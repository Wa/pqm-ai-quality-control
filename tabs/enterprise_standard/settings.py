"""Configuration helpers for the enterprise standard check tab."""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import CONFIG

TAB_ENV_PREFIX = "ENTERPRISE_STANDARD_CHECK"
TAB_SLUG = "enterprise"
KB_MODEL_ID = 7


@dataclass(frozen=True)
class BishengSettings:
    """Resolved configuration values for Bisheng workflow integration."""

    base_url: str
    invoke_path: str
    stop_path: str
    workflow_id: str
    flow_id: str
    flow_input_node_id: str
    flow_milvus_node_id: str
    flow_es_node_id: str
    api_key: str
    max_words: int
    timeout_s: int


def _bisheng_setting(
    name: str,
    *,
    tab_key: str | None = None,
    config_key: str | None = None,
    default: str | None = None,
) -> str | None:
    """Resolve Bisheng settings with tab-specific env and config fallbacks."""

    env_tab_name = f"{TAB_ENV_PREFIX}_{name}"
    env_tab_value = os.getenv(env_tab_name)
    if env_tab_value not in (None, ""):
        return env_tab_value

    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value

    bisheng_config = CONFIG.get("bisheng", {})
    if tab_key:
        tab_value = bisheng_config.get("tabs", {}).get("enterprise_standard_check", {}).get(tab_key)
        if tab_value not in (None, ""):
            return tab_value

    if config_key:
        config_value = bisheng_config.get(config_key)
        if config_value not in (None, ""):
            return config_value

    return default


def _safe_int(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def get_bisheng_settings() -> BishengSettings:
    """Return fully resolved Bisheng configuration for the tab."""

    base_url = _bisheng_setting("BISHENG_BASE_URL", config_key="base_url", default="http://localhost:3001") or ""
    invoke_path = _bisheng_setting("BISHENG_INVOKE_PATH", config_key="invoke_path", default="/api/v2/workflow/invoke") or ""
    stop_path = _bisheng_setting("BISHENG_STOP_PATH", config_key="stop_path", default="/api/v2/workflow/stop") or ""

    workflow_id = _bisheng_setting(
        "BISHENG_WORKFLOW_ID",
        tab_key="workflow_id",
        config_key="workflow_id",
        default="",
    ) or ""
    flow_id = _bisheng_setting("BISHENG_FLOW_ID", tab_key="flow_id", default="") or ""
    flow_input_node_id = _bisheng_setting("FLOW_INPUT_NODE_ID", tab_key="flow_input_node_id", default="") or ""
    flow_milvus_node_id = _bisheng_setting("FLOW_MILVUS_NODE_ID", tab_key="flow_milvus_node_id", default="") or ""
    flow_es_node_id = _bisheng_setting("FLOW_ES_NODE_ID", tab_key="flow_es_node_id", default="") or ""
    api_key = _bisheng_setting("BISHENG_API_KEY", config_key="api_key", default="") or ""

    max_words = _safe_int(
        _bisheng_setting("BISHENG_MAX_WORDS", config_key="max_words", default="2000"),
        2000,
    )
    timeout_s = _safe_int(
        _bisheng_setting("BISHENG_TIMEOUT_S", config_key="timeout_s", default="90"),
        90,
    )

    return BishengSettings(
        base_url=base_url,
        invoke_path=invoke_path,
        stop_path=stop_path,
        workflow_id=workflow_id,
        flow_id=flow_id,
        flow_input_node_id=flow_input_node_id,
        flow_milvus_node_id=flow_milvus_node_id,
        flow_es_node_id=flow_es_node_id,
        api_key=api_key,
        max_words=max_words,
        timeout_s=timeout_s,
    )


__all__ = [
    "BishengSettings",
    "KB_MODEL_ID",
    "TAB_ENV_PREFIX",
    "TAB_SLUG",
    "get_bisheng_settings",
]
