use std::{path::Path, process::Stdio, sync::Arc};

use agent_client_protocol as acp;
use async_trait::async_trait;
use derivative::Derivative;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;
use ts_rs::TS;
use workspace_utils::{command_ext::GroupSpawnNoWindowExt, msg_store::MsgStore};

use crate::{
    approvals::ExecutorApprovalService,
    command::{CmdOverrides, CommandBuildError, CommandBuilder, apply_overrides},
    env::ExecutionEnv,
    executor_discovery::ExecutorDiscoveredOptions,
    executors::{
        AppendPrompt, AvailabilityInfo, BaseCodingAgent, ExecutorError, SpawnedChild,
        StandardCodingAgentExecutor,
        acp::{self as acp_executors, AcpEvent},
    },
    logs::utils::patch,
    model_selector::{ModelSelectorConfig, PermissionPolicy},
    profile::ExecutorConfig,
};

/// Grok has no ACP server mode, but its headless `--output-format streaming-json`
/// emits NDJSON with ACP-shaped session updates (`tool_call`, `tool_call_update`, ...)
/// in its own envelope. This maps one such line onto the internal ACP event stream so
/// the standard ACP normalization pipeline can be reused.
fn grok_line_to_acp_event(line: &str) -> Option<AcpEvent> {
    let mut value: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    let obj = value.as_object_mut()?;
    let event_type = obj.remove("type")?.as_str()?.to_string();
    match event_type.as_str() {
        "text" => {
            let text = obj.remove("data")?.as_str()?.to_string();
            let content = serde_json::from_value(json!({"type": "text", "text": text})).ok()?;
            Some(AcpEvent::Message(content))
        }
        "tool_call" => serde_json::from_value::<acp::ToolCall>(value)
            .ok()
            .map(AcpEvent::ToolCall),
        "tool_call_update" => serde_json::from_value::<acp::ToolCallUpdate>(value)
            .ok()
            .map(AcpEvent::ToolUpdate),
        // `end` carries the session id; SessionStart makes the pipeline store it for follow-ups
        "end" => obj
            .remove("sessionId")
            .and_then(|v| v.as_str().map(str::to_string))
            .map(AcpEvent::SessionStart),
        _ => None,
    }
}

#[derive(Derivative, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[derivative(Debug, PartialEq)]
pub struct Grok {
    #[serde(default)]
    pub append_prompt: AppendPrompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        title = "Bypass Permissions",
        description = "Run all tool calls without asking for approval."
    )]
    pub bypass_permissions: Option<bool>,
    #[serde(flatten)]
    pub cmd: CmdOverrides,
    #[serde(skip)]
    #[ts(skip)]
    #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub approvals: Option<Arc<dyn ExecutorApprovalService>>,
}

impl Grok {
    fn build_command_builder(&self) -> Result<CommandBuilder, CommandBuildError> {
        let mut builder =
            CommandBuilder::new("grok").extend_params(["--output-format", "streaming-json"]);

        if let Some(model) = &self.model {
            builder = builder.extend_params(["--model", model.as_str()]);
        }

        if self.bypass_permissions.unwrap_or(false) {
            builder = builder.extend_params(["--permission-mode", "bypassPermissions"]);
        }
        apply_overrides(builder, &self.cmd)
    }

    async fn spawn_grok(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_args: &[String],
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let command_parts = self.build_command_builder()?.build_initial()?;
        let (executable_path, args) = command_parts.into_resolved().await?;

        let combined_prompt = self.append_prompt.combine_prompt(prompt);

        let mut command = Command::new(executable_path);
        command
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(current_dir)
            .args(&args)
            .args(session_args)
            .arg("--single")
            .arg(&combined_prompt);

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut command);

        Ok(command.group_spawn_no_window()?.into())
    }
}

#[async_trait]
impl StandardCodingAgentExecutor for Grok {
    fn apply_overrides(&mut self, executor_config: &ExecutorConfig) {
        if let Some(model_id) = executor_config.model_id.as_ref() {
            self.model = Some(model_id.clone());
        }
        if let Some(permission_policy) = executor_config.permission_policy.clone() {
            self.bypass_permissions = Some(matches!(
                permission_policy,
                crate::model_selector::PermissionPolicy::Auto
            ));
        }
    }

    fn use_approvals(&mut self, approvals: Arc<dyn ExecutorApprovalService>) {
        self.approvals = Some(approvals);
    }

    async fn spawn(
        &self,
        current_dir: &Path,
        prompt: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        // Grok only accepts a caller-chosen id for new sessions, so we mint one here;
        // follow-ups then resume it with `--resume`.
        let session_id = uuid::Uuid::new_v4().to_string();
        self.spawn_grok(
            current_dir,
            prompt,
            &["--session-id".to_string(), session_id],
            env,
        )
        .await
    }

    async fn spawn_follow_up(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: &str,
        _reset_to_message_id: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        self.spawn_grok(
            current_dir,
            prompt,
            &["--resume".to_string(), session_id.to_string()],
            env,
        )
        .await
    }

    fn normalize_logs(
        &self,
        msg_store: Arc<MsgStore>,
        worktree_path: &Path,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        acp_executors::normalize_logs_with_parser(
            msg_store,
            worktree_path,
            &[],
            grok_line_to_acp_event,
        )
    }

    fn default_mcp_config_path(&self) -> Option<std::path::PathBuf> {
        // MCP не настраиваем через VK: grok управляет своими MCP в ~/.grok/config.toml
        None
    }

    fn get_availability_info(&self) -> AvailabilityInfo {
        let installation_found = dirs::home_dir()
            .map(|home| home.join(".grok").exists())
            .unwrap_or(false);

        if installation_found {
            AvailabilityInfo::InstallationFound
        } else {
            AvailabilityInfo::NotFound
        }
    }

    fn get_preset_options(&self) -> ExecutorConfig {
        ExecutorConfig {
            executor: BaseCodingAgent::Grok,
            variant: None,
            model_id: self.model.clone(),
            agent_id: None,
            reasoning_id: None,
            permission_policy: Some(if self.bypass_permissions.unwrap_or(false) {
                PermissionPolicy::Auto
            } else {
                PermissionPolicy::Supervised
            }),
        }
    }

    async fn discover_options(
        &self,
        _workdir: Option<&std::path::Path>,
        _repo_path: Option<&std::path::Path>,
    ) -> Result<futures::stream::BoxStream<'static, json_patch::Patch>, ExecutorError> {
        let options = ExecutorDiscoveredOptions {
            model_selector: ModelSelectorConfig {
                permissions: vec![PermissionPolicy::Auto, PermissionPolicy::Supervised],
                ..Default::default()
            },
            ..Default::default()
        };
        Ok(Box::pin(futures::stream::once(async move {
            patch::executor_discovered_options(options)
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_grok_end_line_into_session_start() {
        let line = r#"{"type":"end","stopReason":"end_turn","sessionId":"abc-123"}"#;
        let event = grok_line_to_acp_event(line);
        assert!(matches!(event, Some(AcpEvent::SessionStart(id)) if id == "abc-123"));
    }

    #[test]
    fn parses_grok_tool_call_line() {
        let line = r#"{"type":"tool_call","toolCallId":"call-1","title":"write","kind":"write","status":"pending","toolName":"write","rawInput":{"file_path":"/tmp/x"},"content":[],"locations":[]}"#;
        let event = grok_line_to_acp_event(line);
        assert!(matches!(event, Some(AcpEvent::ToolCall(_))));
    }

    #[test]
    fn ignores_unknown_lines() {
        assert!(grok_line_to_acp_event(r#"{"type":"usage","usage":{}}"#).is_none());
        assert!(grok_line_to_acp_event("not json").is_none());
    }
}
