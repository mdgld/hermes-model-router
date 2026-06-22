"""
model-router plugin
===================
Automatic cost-aware model routing.

Runtime source of truth:
  - active profile `model_router.yaml`

Built-in defaults only apply when that file does not exist yet or is invalid.

Features:
  - Pre-LLM classification on every turn
  - Explicit tier request detection: T3, tier4, t2, etc.
  - /model and /t1-/t5 set a session pin until /auto or a fresh session
  - Mid-loop self-escalation on repeated tool errors
  - Post-heavy-work de-escalation after T4/T5 completes
  - Status bar: shows [Tx] prefix dynamically in front of model name
  - Ambiguous multi-tier mention guard: "T1 T2 T3" = discussing tiers, not requesting one
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model definitions used only for bootstrap/fallback
# KEEP IN SYNC with install.py DEFAULT_ROUTER_CONFIG — duplicated intentionally
# so install.py remains a self-contained script with no import dependencies.
# ---------------------------------------------------------------------------

DEFAULT_ROUTER_CONFIG = {
    "provider_priority": ["nous", "openai-codex", "openrouter"],
    "classifier": {
        "provider": "nous",
        "model": "deepseek/deepseek-v4-flash",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "api_key": "",
        "timeout": 30,
        "extra_body": {"enable_caching": True, "reasoning_effort": "high"},
        "fallbacks": [
            {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning_effort": "low"},
            {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash", "reasoning_effort": "high"},
        ],
    },
    "tiers": {
        1: {
            "label": "T1 Flash (MiMo v2.5 Pro)",
            "emoji": "⚡",
            "model": "xiaomi/mimo-v2.5-pro",
            "reasoning": "enabled",
            "extra_body": {"enable_caching": True},
            "role": "fast triage and cheap helper",
            "best_for": [
                "Short acknowledgements",
                "Intent classification",
                "Status checks",
                "Title generation",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning": "low", "extra_body": {"enable_caching": True}},
                {"provider": "nous", "model": "deepseek/deepseek-v4-flash", "reasoning_effort": "high", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "xiaomi/mimo-v2.5-pro", "reasoning": "enabled", "extra_body": {"enable_caching": True}},
            ],
        },
        2: {
            "label": "T2 (DeepSeek v4 Pro)",
            "emoji": "🔹",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning": "xhigh",
            "extra_body": {"enable_caching": True},
            "role": "day-to-day usage, basic tasks",
            "best_for": [
                "Standard day-to-day work",
                "Well-defined documentation and drafting",
                "Extremely basic coding and research",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4-mini", "reasoning": "medium", "extra_body": {"enable_caching": True}},
                {"provider": "nous", "model": "deepseek/deepseek-v4-flash", "reasoning_effort": "high", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "reasoning": "xhigh", "extra_body": {"enable_caching": True}},
            ],
        },
        3: {
            "label": "T3 (MiniMax M3)",
            "emoji": "🔷",
            "model": "minimax/minimax-m3",
            "reasoning": "enabled",
            "extra_body": {"enable_caching": True},
            "role": "standard coding, well-defined short tasks",
            "best_for": [
                "Basic troubleshooting",
                "Light code review",
                "Standard reasoning",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4", "reasoning": "xhigh", "extra_body": {"enable_caching": True}},
                {"provider": "nous", "model": "deepseek/deepseek-v4-pro", "reasoning_effort": "high", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "minimax/minimax-m3", "reasoning": "enabled", "extra_body": {"enable_caching": True}},
            ],
        },
        4: {
            "label": "T4 (GLM 5.2)",
            "emoji": "🔸",
            "model": "z-ai/glm-5.2",
            "reasoning": "xhigh",
            "extra_body": {"enable_caching": True},
            "role": "strong reasoning and synthesis",
            "best_for": [
                "Architecture and refactoring",
                "Migration planning",
                "Basic agentic workflows",
                "Complex multi-step designs and workflows",
                "Nuanced code review",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.4", "reasoning": "high", "extra_body": {"enable_caching": True}},
                {"provider": "nous", "model": "z-ai/glm-5.2", "reasoning_effort": "xhigh", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "z-ai/glm-5.2", "reasoning": "xhigh", "extra_body": {"enable_caching": True}},
            ],
        },
        5: {
            "label": "T5 (GPT-5.5)",
            "emoji": "🔶🔶",
            "model": "openai/gpt-latest",
            "reasoning": "high",
            "extra_body": {"enable_caching": True},
            "role": "expensive deep-think mode",
            "best_for": [
                "Security-sensitive analysis",
                "High-stakes tasks",
                "Algorithmic optimization",
                "Long-context agentic workflows",
                "Near-human-level reasoning on certain tasks",
            ],
            "fallbacks": [
                {"provider": "openai-codex", "model": "gpt-5.5", "reasoning": "high"},
                {"provider": "nous", "model": "z-ai/glm-5.2", "reasoning_effort": "high", "extra_body": {"enable_caching": True}},
                {"provider": "openrouter", "model": "openai/gpt-latest", "reasoning": "high", "extra_body": {"enable_caching": True}},
            ],
        },
    },
    "task_routes": [
        {
            "name": "security",
            "tier": 5,
            "working_tier": 5,
            "floor_tier": 4,
            "priority": 100,
            "keywords": ["security", "vulnerability", "exploit", "auth", "secret", "threat model"],
        },
        {
            "name": "algorithmic_optimization",
            "tier": 5,
            "working_tier": 5,
            "floor_tier": 3,
            "priority": 95,
            "keywords": ["algorithm", "complexity", "optimize", "optimization", "benchmark", "performance"],
        },
        {
            "name": "architecture_planning",
            "tier": 4,
            "working_tier": 4,
            "floor_tier": 3,
            "priority": 80,
            "keywords": ["architecture", "design", "migration", "refactor", "workflow", "plan"],
        },
        {
            "name": "debugging",
            "tier": 2,
            "working_tier": 2,
            "floor_tier": 1,
            "priority": 75,
            "keywords": ["debug", "bug", "root cause", "traceback", "stack trace", "failing test", "regression", "investigate"],
        },
        {
            "name": "normal_coding",
            "tier": 1,
            "working_tier": 1,
            "floor_tier": 1,
            "priority": 55,
            "keywords": ["implement", "add function", "write code", "build", "feature", "endpoint"],
        },
        {
            "name": "quick_edit",
            "tier": 1,
            "working_tier": 1,
            "floor_tier": 1,
            "priority": 50,
            "keywords": ["rename", "typo", "tweak", "one-liner", "small fix", "adjust", "bump"],
        },
        {
            "name": "drafting_summary",
            "tier": 1,
            "working_tier": 1,
            "floor_tier": 1,
            "priority": 40,
            "keywords": ["draft", "rewrite", "summarize", "summary", "email", "docs", "documentation"],
        },
    ],
    "default_floor_delta": 2,
}

_router_config: dict[str, Any] = copy.deepcopy(DEFAULT_ROUTER_CONFIG)
TIERS: dict[int, dict[str, Any]] = {}
RUNTIME_PROFILES: dict[str, dict[str, Any]] = {}
MODEL_TO_TIER: dict[str, int] = {}
FLASH_MODEL = ""
FLASH_PROVIDER = ""
_TIER_LABELS: dict[int, tuple[str, str]] = {}
PROVIDER_PRIORITY: list[str] = []
TIER_FALLBACKS: dict[int, list[dict[str, Any]]] = {}
CLASSIFIER_FALLBACKS: list[dict[str, Any]] = []
TASK_ROUTES: list[dict[str, Any]] = []
_provider_failures: dict[str, float] = {}
_session_runtime_state: dict[str, dict[str, Any]] = {}
_session_route_trace: dict[str, dict[str, Any]] = {}
_startup_validation_status: dict[str, Any] = {}
_PROVIDER_UNHEALTHY_TTL = 120.0


def _normalize_task_routes(raw_routes: list) -> list[dict[str, Any]]:
    """Validate, normalize, and sort task routes by descending priority."""
    if not isinstance(raw_routes, list):
        return []
    normalized = []
    for route in raw_routes:
        if not isinstance(route, dict):
            continue
        name = str(route.get("name") or "").strip()
        # working_tier is canonical; "tier" is the back-compat alias
        working_tier = route.get("working_tier") or route.get("tier")
        priority = route.get("priority", 50)
        keywords = route.get("keywords", [])
        if not name or not isinstance(working_tier, int) or working_tier not in range(1, 6):
            continue
        if not isinstance(keywords, list) or not keywords:
            continue
        floor_raw = route.get("floor_tier", working_tier)
        floor_tier = int(floor_raw) if isinstance(floor_raw, (int, float)) else working_tier
        floor_tier = max(1, min(floor_tier, working_tier))  # clamp: floor ∈ [1, working_tier]
        normalized.append({
            "name": name,
            "tier": int(working_tier),          # back-compat for callers that only look at ["tier"]
            "working_tier": int(working_tier),
            "floor_tier": floor_tier,
            "priority": int(priority) if isinstance(priority, (int, float)) else 50,
            "keywords": [str(k).strip().lower() for k in keywords if str(k).strip()],
        })
    normalized.sort(key=lambda r: r["priority"], reverse=True)
    return normalized


_PROVIDER_BASE_URLS: dict[str, str] = {
    "nous": "https://inference-api.nousresearch.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
}

_PROVIDER_SHORT: dict[str, str] = {
    "bedrock": "aws",
    "openrouter": "or",
    "openai-codex": "codex",
    "openai": "oai",
    "nous": "nous",
    "anthropic": "anth",
    "deepseek": "ds",
}

# Per-provider reasoning effort normalisation.
# "max" is Bedrock-internal; bedrock_adapter.py already maps xhigh→max natively so
# bedrock is left as a no-op here.  All other providers accept the canonical OpenAI
# set: none|minimal|low|medium|high|xhigh (model-dependent per OpenAI docs).
_PROVIDER_REASONING_NORM: dict[str, dict[str, str]] = {
    "openrouter":   {"max": "xhigh", "enabled": "high",   "adaptive": "xhigh"},
    "nous":         {"max": "xhigh", "enabled": "high",   "adaptive": "xhigh"},
    "openai":       {"max": "xhigh", "enabled": "medium", "adaptive": "medium"},
    "openai-codex": {"max": "xhigh", "enabled": "medium", "adaptive": "medium"},
    "anthropic":    {"max": "high",  "enabled": "medium", "adaptive": "medium"},
    "deepseek":     {"max": "high",  "xhigh": "high",     "enabled": "high",   "adaptive": "high"},
    "bedrock":      {},
}


def _normalize_reasoning_for_provider(effort: str | None, provider: str) -> str | None:
    if not effort or not provider:
        return effort
    p = str(provider).strip().lower()
    mapping = _PROVIDER_REASONING_NORM.get(p, {"max": "xhigh", "enabled": "medium"})
    return mapping.get(effort, effort)


def _get_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _router_config_path() -> Path:
    return _get_hermes_home() / "model_router.yaml"


def _router_state_dir() -> Path:
    return _get_hermes_home() / "model-router"


def _router_state_path() -> Path:
    return _router_state_dir() / "state.json"


def _router_events_path() -> Path:
    return _router_state_dir() / "events.jsonl"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _determine_api_mode(provider: str, base_url: str) -> str:
    try:
        from hermes_cli.providers import determine_api_mode  # type: ignore

        return determine_api_mode(provider, base_url)
    except Exception:
        return "bedrock_converse" if provider == "bedrock" else "chat_completions"


def _normalize_runtime_profile(
    profile_id: str,
    raw: dict[str, Any] | None,
    tier_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    tier_meta = tier_meta if isinstance(tier_meta, dict) else {}

    provider = str(raw.get("provider") or tier_meta.get("provider") or "").strip()
    wire_model = str(
        raw.get("wire_model")
        or raw.get("model")
        or tier_meta.get("model")
        or ""
    ).strip()
    base_url = str(
        raw.get("base_url")
        or tier_meta.get("base_url")
        or _PROVIDER_BASE_URLS.get(provider, "")
    ).strip()
    api_mode = str(
        raw.get("api_mode")
        or tier_meta.get("api_mode")
        or _determine_api_mode(provider, base_url)
    ).strip()
    request_policy = raw.get("request_policy")
    if not isinstance(request_policy, dict):
        request_policy = {}

    capability_tags = raw.get("capability_tags")
    if not isinstance(capability_tags, list):
        capability_tags = []

    fallback_profiles = raw.get("fallback_profiles")
    if not isinstance(fallback_profiles, list):
        fallback_profiles = []

    return {
        "profile_id": profile_id,
        "display_name": str(
            raw.get("display_name")
            or tier_meta.get("label")
            or profile_id
        ),
        "provider": provider,
        "wire_model": wire_model,
        "model": wire_model,
        "base_url": base_url,
        "api_mode": api_mode,
        "reasoning_mode": str(raw.get("reasoning_mode") or tier_meta.get("reasoning_mode") or "").strip(),
        "auth_source": str(raw.get("auth_source") or "").strip(),
        "reasoning": raw.get("reasoning", tier_meta.get("reasoning")),
        "request_policy": request_policy,
        "fallback_profiles": fallback_profiles,
        "capability_tags": capability_tags,
    }


def _normalize_router_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    merged = _deep_merge(DEFAULT_ROUTER_CONFIG, raw or {})
    normalized_tiers: dict[int, dict[str, Any]] = {}
    raw_tiers = merged.get("tiers", {})
    for tier_num in range(1, 6):
        tier_defaults = copy.deepcopy(DEFAULT_ROUTER_CONFIG["tiers"][tier_num])
        override = raw_tiers.get(tier_num, raw_tiers.get(str(tier_num), {}))
        if not isinstance(override, dict):
            override = {}
        tier_defaults.update(override)
        # Preserve fallbacks from user yaml if present; they may differ from defaults
        if "fallbacks" in override:
            tier_defaults["fallbacks"] = override["fallbacks"]
        normalized_tiers[tier_num] = tier_defaults
    merged["tiers"] = normalized_tiers

    raw_runtime_profiles = merged.get("runtime_profiles", {})
    if not isinstance(raw_runtime_profiles, dict):
        raw_runtime_profiles = {}

    runtime_profiles: dict[str, dict[str, Any]] = {}
    for profile_id, raw_profile in raw_runtime_profiles.items():
        if isinstance(raw_profile, dict):
            runtime_profiles[str(profile_id)] = _normalize_runtime_profile(
                str(profile_id),
                raw_profile,
            )

    for tier_num, tier_defaults in normalized_tiers.items():
        profile_id = str(
            tier_defaults.get("target")
            or tier_defaults.get("runtime_profile")
            or f"tier_{tier_num}"
        )
        tier_defaults["target"] = profile_id
        runtime_profiles[profile_id] = _normalize_runtime_profile(
            profile_id,
            raw_runtime_profiles.get(profile_id),
            tier_defaults,
        )

    merged["runtime_profiles"] = runtime_profiles
    return merged


def _apply_router_config(config: dict[str, Any]) -> None:
    global _router_config, TIERS, RUNTIME_PROFILES, MODEL_TO_TIER, FLASH_MODEL, FLASH_PROVIDER, _TIER_LABELS
    global PROVIDER_PRIORITY, TIER_FALLBACKS, CLASSIFIER_FALLBACKS, TASK_ROUTES

    _router_config = config
    RUNTIME_PROFILES = copy.deepcopy(config.get("runtime_profiles", {}))
    TIERS = {
        tier_num: {
            "model": meta["model"],
            "provider": meta.get("provider"),
            "reasoning": meta.get("reasoning"),
            "label": meta.get("label", f"T{tier_num}"),
            "emoji": meta.get("emoji", ""),
            "role": meta.get("role", ""),
            "best_for": meta.get("best_for", []),
            "target": meta.get("target"),
        }
        for tier_num, meta in config["tiers"].items()
    }

    model_to_tier: dict[str, int] = {}
    for tier_num in sorted(TIERS):
        model = TIERS[tier_num]["model"]
        model_to_tier.setdefault(model, tier_num)
        profile_id = str(TIERS[tier_num].get("target") or "")
        runtime_profile = RUNTIME_PROFILES.get(profile_id, {})
        runtime_model = str(runtime_profile.get("wire_model") or "")
        if runtime_model:
            model_to_tier.setdefault(runtime_model, tier_num)
    MODEL_TO_TIER = model_to_tier

    classifier = config.get("classifier", {})
    FLASH_MODEL = classifier.get("model", DEFAULT_ROUTER_CONFIG["classifier"]["model"])
    FLASH_PROVIDER = classifier.get("provider", DEFAULT_ROUTER_CONFIG["classifier"]["provider"])

    _TIER_LABELS = {
        tier_num: (meta.get("emoji", ""), meta.get("label", f"T{tier_num}"))
        for tier_num, meta in config["tiers"].items()
    }

    PROVIDER_PRIORITY = config.get("provider_priority", list(DEFAULT_ROUTER_CONFIG.get("provider_priority", [])))
    CLASSIFIER_FALLBACKS = classifier.get("fallbacks", list(DEFAULT_ROUTER_CONFIG["classifier"].get("fallbacks", [])))
    TIER_FALLBACKS = {
        tier_num: list(meta.get("fallbacks", []))
        for tier_num, meta in config["tiers"].items()
    }

    raw_task_routes = config.get("task_routes", DEFAULT_ROUTER_CONFIG.get("task_routes", []))
    TASK_ROUTES = _normalize_task_routes(raw_task_routes)


def _tier_profile_id(tier: int) -> str:
    return str(TIERS.get(tier, {}).get("target") or f"tier_{tier}")


def _runtime_bundle_from_profile(profile_id: str) -> dict[str, Any]:
    profile = dict(RUNTIME_PROFILES.get(profile_id, {}))
    if not profile:
        return {}
    profile.setdefault("profile_id", profile_id)
    profile.setdefault("model", profile.get("wire_model", ""))
    return profile


def resolve_tier_runtime(tier: int) -> dict[str, Any]:
    """Return the resolved runtime bundle for a tier.

    The router resolves to a provider-specific runtime profile rather than only a
    model slug so gateway overrides and live switches stay provider-correct.
    """
    tier_data = TIERS.get(tier, {})
    profile_id = _tier_profile_id(tier)
    primary = _runtime_bundle_from_profile(profile_id)
    primary_provider = str(primary.get("provider") or "").strip()
    primary_model = str(primary.get("model") or "").strip()

    if primary_model and _is_provider_healthy(primary_provider):
        return primary

    for idx, fb in enumerate(TIER_FALLBACKS.get(tier, []), start=1):
        fb_provider = str(fb.get("provider") or "").strip()
        fb_model = str(fb.get("wire_model") or fb.get("model") or "").strip()
        if not fb_provider or not fb_model or not _is_provider_healthy(fb_provider):
            continue
        logger.info(
            "model-router: T%d primary provider %r unhealthy; using fallback %r/%s",
            tier, primary_provider, fb_provider, fb_model,
        )
        fb_base_url = str(fb.get("base_url") or _PROVIDER_BASE_URLS.get(fb_provider, "")).strip()
        return {
            "profile_id": f"{profile_id}::fallback::{idx}",
            "display_name": str(tier_data.get("label") or f"T{tier}"),
            "provider": fb_provider,
            "wire_model": fb_model,
            "model": fb_model,
            "base_url": fb_base_url,
            "api_mode": str(fb.get("api_mode") or _determine_api_mode(fb_provider, fb_base_url)).strip(),
            "reasoning_mode": str(fb.get("reasoning_mode") or "").strip(),
            "auth_source": str(fb.get("auth_source") or "").strip(),
            "reasoning": fb.get("reasoning", primary.get("reasoning")),
            "request_policy": dict(fb.get("request_policy") or {}),
            "fallback_profiles": list(fb.get("fallback_profiles") or []),
            "capability_tags": list(fb.get("capability_tags") or []),
        }

    return primary


def validate_router_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate router config structure. Returns {valid, warnings, errors}."""
    errors: list[str] = []
    warnings: list[str] = []

    tiers = config.get("tiers", {})
    runtime_profiles = config.get("runtime_profiles", {})

    for tier_num in range(1, 6):
        tier = tiers.get(tier_num, tiers.get(str(tier_num), {}))
        if not isinstance(tier, dict):
            errors.append(f"tier {tier_num}: missing or invalid")
            continue
        model = str(tier.get("model") or "").strip()
        if not model:
            errors.append(f"tier {tier_num}: model is empty")
        profile_id = str(tier.get("target") or f"tier_{tier_num}")
        profile = runtime_profiles.get(profile_id, {})
        if profile:
            wire_model = str(profile.get("wire_model") or profile.get("model") or "").strip()
            provider = str(profile.get("provider") or "").strip()
            api_mode = str(profile.get("api_mode") or "").strip()
            if not wire_model:
                warnings.append(f"runtime profile {profile_id!r}: wire_model is empty")
            if not provider:
                warnings.append(f"runtime profile {profile_id!r}: provider is empty")
            if not api_mode:
                warnings.append(f"runtime profile {profile_id!r}: api_mode is empty")

    for pid, profile in runtime_profiles.items():
        if not isinstance(profile, dict):
            errors.append(f"runtime profile {pid!r}: not a dict")

    valid = len(errors) == 0
    return {"valid": valid, "errors": errors, "warnings": warnings}


def get_router_startup_status() -> dict[str, Any]:
    return dict(_startup_validation_status)


def get_router_analytics(limit: int = 100) -> dict[str, Any]:
    """Aggregate routing metrics from events.jsonl. Bounded by limit."""
    try:
        max_limit = max(1, min(int(limit or 100), 500))
    except Exception:
        max_limit = 100
    events = get_recent_events("", limit=max_limit)
    tier_counts: dict[int, int] = {}
    reason_counts: dict[str, int] = {}
    task_route_hits: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    classifier_fallback_count = 0
    mismatch_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event") or "")
        if event_type == "route_decision":
            tier = event.get("tier")
            reason = str(event.get("reason") or "")
            provider = str(event.get("provider") or "")
            route_name = str(event.get("route_name") or "")
            if isinstance(tier, (int, float)) and 1 <= int(tier) <= 5:
                k = int(tier)
                tier_counts[k] = tier_counts.get(k, 0) + 1
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason == "task_route" and route_name:
                task_route_hits[route_name] = task_route_hits.get(route_name, 0) + 1
            if provider:
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
        elif event_type == "classifier_fallback":
            classifier_fallback_count += 1
        elif event_type in ("mismatch", "router_override_mismatch"):
            mismatch_count += 1
    return {
        "total_events_read": len(events),
        "tier_counts": tier_counts,
        "reason_counts": reason_counts,
        "task_route_hits": task_route_hits,
        "provider_counts": provider_counts,
        "classifier_fallback_count": classifier_fallback_count,
        "mismatch_count": mismatch_count,
    }


_DEFAULT_EVAL_FIXTURES: list[dict[str, Any]] = [
    {"prompt": "security vulnerability in this auth module", "expected_tier": 5, "expected_reason": "task_route"},
    {"prompt": "optimize performance of this algorithm", "expected_tier": 5, "expected_reason": "task_route"},
    {"prompt": "architecture for this migration plan", "expected_tier": 4, "expected_reason": "task_route"},
    {"prompt": "debug this failing test root cause", "expected_tier": 3, "expected_reason": "task_route"},
    {"prompt": "T3 quick question about code", "expected_tier": 3, "expected_reason": "explicit_tier"},
    {"prompt": "ok", "expected_tier": 1, "expected_reason": "ack"},
    {"prompt": "thanks", "expected_tier": 1, "expected_reason": "ack"},
]


def eval_task_routing(fixtures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Deterministically evaluate task routing without calling external LLMs.

    Each fixture: {prompt, expected_tier, expected_reason}.
    Returns: {total, passed, failed, results: [{prompt, expected_tier, expected_reason, actual_tier, actual_reason, pass}]}.
    """
    if fixtures is None:
        fixtures = list(_DEFAULT_EVAL_FIXTURES)
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        prompt = str(fixture.get("prompt") or "")
        expected_tier = int(fixture.get("expected_tier") or 0)
        expected_reason = str(fixture.get("expected_reason") or "")
        actual_reason = "classifier"
        actual_tier = 0
        explicit = _detect_explicit_tier(prompt)
        if explicit is not None:
            actual_tier = explicit
            actual_reason = "explicit_tier"
        elif _is_obvious_ack(prompt):
            actual_tier = 1
            actual_reason = "ack"
        else:
            route_match = _match_task_route(prompt)
            if route_match is not None:
                actual_tier = route_match["tier"]
                actual_reason = "task_route"
        is_pass = (actual_tier == expected_tier and actual_reason == expected_reason)
        if is_pass:
            passed += 1
        else:
            failed += 1
        results.append({
            "prompt": prompt[:80],
            "expected_tier": expected_tier,
            "expected_reason": expected_reason,
            "actual_tier": actual_tier,
            "actual_reason": actual_reason,
            "pass": is_pass,
        })
    return {"total": len(results), "passed": passed, "failed": failed, "results": results}


def _load_router_config() -> None:
    global _startup_validation_status
    path = _router_config_path()
    if not path.exists():
        _apply_router_config(copy.deepcopy(DEFAULT_ROUTER_CONFIG))
        validation = validate_router_config(_router_config)
        _startup_validation_status = validation
        return

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("model_router.yaml must be a mapping")
        normalized = _normalize_router_config(raw)
        _apply_router_config(normalized)
        validation = validate_router_config(_router_config)
        _startup_validation_status = validation
        if not validation["valid"]:
            for err in validation["errors"]:
                logger.error("model-router: config error: %s", err)
        for warn in validation["warnings"]:
            logger.warning("model-router: config warning: %s", warn)
        logger.info("model-router: loaded config from %s", path)
    except Exception as exc:
        logger.warning("model-router: failed to load %s: %s -- using defaults", path, exc)
        _apply_router_config(copy.deepcopy(DEFAULT_ROUTER_CONFIG))
        _startup_validation_status = {"valid": False, "errors": [str(exc)], "warnings": []}


def _persist_router_state() -> None:
    try:
        with _state_lock:
            payload = {
                "version": 1,
                "session_manual": dict(_session_manual),
                "session_pinned": dict(_session_pinned),
                "last_tier": dict(_last_tier),
                "base_tier": dict(_base_tier),
                "session_runtime_state": dict(_session_runtime_state),
                "session_working": dict(_session_working),
                "session_floor": dict(_session_floor),
                "mechanical_streak": dict(_mechanical_streak),
            }
        path = _router_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception as exc:
        logger.debug("model-router: failed to persist state: %s", exc)


def _append_router_event(
    event_type: str,
    session_id: str = "",
    **fields: Any,
) -> None:
    try:
        payload = {
            "ts": time.time(),
            "event": str(event_type or "").strip(),
            "session_id": str(session_id or "").strip(),
        }
        for key, value in fields.items():
            if value is None:
                continue
            payload[str(key)] = value
        path = _router_events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception as exc:
        logger.debug("model-router: failed to append event %r: %s", event_type, exc)


def get_recent_events(session_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
    path = _router_events_path()
    if not path.exists():
        return []
    try:
        max_items = max(1, min(int(limit or 20), 100))
    except Exception:
        max_items = 20
    session_filter = str(session_id or "").strip()
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                if session_filter and str(item.get("session_id") or "") != session_filter:
                    continue
                events.append(item)
        return events[-max_items:]
    except Exception as exc:
        logger.debug("model-router: failed to read events: %s", exc)
        return []


def _load_persisted_state() -> None:
    path = _router_state_path()
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return
        with _state_lock:
            for key, target in (
                ("session_manual", _session_manual),
                ("session_pinned", _session_pinned),
                ("last_tier", _last_tier),
                ("base_tier", _base_tier),
                ("session_runtime_state", _session_runtime_state),
                ("session_working", _session_working),
                ("session_floor", _session_floor),
                ("mechanical_streak", _mechanical_streak),
            ):
                incoming = raw.get(key, {})
                if isinstance(incoming, dict):
                    target.clear()
                    target.update(incoming)
        logger.info("model-router: restored persisted state from %s", path)
    except Exception as exc:
        logger.debug("model-router: failed to restore persisted state: %s", exc)


def get_router_diagnostics(session_id: str = "", limit: int = 10) -> dict[str, Any]:
    session_filter = str(session_id or "").strip()
    state = get_session_state(session_filter) if session_filter else {}
    diagnostics = {
        "session_id": session_filter,
        "state": state,
        "recent_events": get_recent_events(session_filter, limit=limit),
    }
    if state:
        diagnostics["pinned"] = bool(state.get("pinned"))
        diagnostics["tier"] = int(state.get("tier") or state.get("last_tier") or 0)
        diagnostics["profile_id"] = str(state.get("profile_id") or "")
        diagnostics["model"] = str(state.get("model") or "")
        diagnostics["provider"] = str(state.get("provider") or "")
        diagnostics["api_mode"] = str(state.get("api_mode") or "")
        diagnostics["route_reason"] = str(state.get("route_reason") or "")
        diagnostics["route_name"] = state.get("route_name")
        diagnostics["route_keyword"] = state.get("route_keyword")
    diagnostics["startup_validation"] = dict(_startup_validation_status)
    return diagnostics


_apply_router_config(copy.deepcopy(DEFAULT_ROUTER_CONFIG))
_startup_validation_status = validate_router_config(_router_config)

# ---------------------------------------------------------------------------
# Explicit tier/model request detection
# ---------------------------------------------------------------------------

# Detect standalone tier mentions: "T3", "t2", "T4", "(T5)", etc.
# Avoids false positives inside words: "cat5", "step3", "T100"
_TIER_STANDALONE_RE = re.compile(
    r"(?:^|(?<=\s)|(?<=\())[tT]([1-5])(?:\b|(?=\)))"
)
_TIER_WORD_RE = re.compile(
    r"(?:^|(?<=\s)|(?<=\())tier\s*([1-5])", re.IGNORECASE
)


def _detect_explicit_tier(msg: str) -> int | None:
    """Return an explicitly requested tier, or None if ambiguous/absent.

    Supported syntax:
      T3, t2, T4, T5      — standalone Tx notation (case-insensitive)
      tier4, tier 3       — word form (case-insensitive, TIER 3, Tier5, etc.)

    Model name keywords (sonnet, deepseek, flash, plus, qwen...) are intentionally
    NOT supported here — use /model for explicit model selection. Those keywords
    cause more false positives than true routing signals.

    Ambiguity guard: if the message mentions 3+ distinct tier numbers
    (e.g. "T1 T2 T3 T4 T5" or discussing the routing table) we treat it
    as an informational mention, not a routing directive, and defer to Flash.
    This prevents the common false positive where the user asks *about* tiers.

    Two mentions: takes the HIGHEST (e.g. "T3 vs T4 approach" → T4).
    """
    # Collect all Tx / tierN mentions
    all_tier_mentions = set(
        int(m) for m in _TIER_STANDALONE_RE.findall(msg)
    )
    word_mentions = set(
        int(m) for m in _TIER_WORD_RE.findall(msg)
    )
    all_tier_mentions |= word_mentions

    # Ambiguity guard: 3+ distinct tier numbers = discussing tiers, not requesting
    if len(all_tier_mentions) >= 3:
        logger.debug(
            "model-router: %d distinct tier mentions in msg -- ambiguous, deferring to Flash",
            len(all_tier_mentions),
        )
        return None

    # Exactly one or two mentions -> take the HIGHEST one as the intent
    # (e.g. "compare T3 vs T4 approach" -> T4 makes sense as a cap)
    if all_tier_mentions:
        tier_num = max(all_tier_mentions)
        logger.info("model-router: explicit tier request detected: T%d", tier_num)
        return tier_num

    return None


# ---------------------------------------------------------------------------
# Fast-path heuristic: obvious short ACKs -> T1, no Flash call
# ---------------------------------------------------------------------------

def _match_task_route(msg: str) -> dict[str, Any] | None:
    """Return first task-route match (highest priority) or None. Never raises."""
    if not TASK_ROUTES or not msg:
        return None
    search_text = msg[:800].lower()
    try:
        for route in TASK_ROUTES:
            for keyword in route["keywords"]:
                if keyword in search_text:
                    return {
                        "name": route["name"],
                        "tier": route["tier"],
                        "working_tier": route.get("working_tier", route["tier"]),
                        "floor_tier": route.get("floor_tier", route["tier"]),
                        "keyword": keyword,
                    }
    except Exception as exc:
        logger.debug("model-router: task route matching error: %s", exc)
    return None


_ACK_RE = re.compile(
    r"^(ok|okay|thanks|thank you|thx|got it|understood|sure|yes|no|yep|nope|"
    r"alright|cool|great|nice|perfect|done|noted|ack|"
    r"dzięki|tak|nie|rozumiem|gotowe|spoko|super|świetnie|"
    r"hello|hi|hey|cześć|hej)"
    r"[!?.]*$",
    re.IGNORECASE,
)

def _is_obvious_ack(msg: str) -> bool:
    words = msg.strip().split()
    if len(words) > 6:
        return False
    return bool(_ACK_RE.match(msg.strip()))


# ---------------------------------------------------------------------------
# Flash classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """\
You are a model routing classifier. Your only job is to assign a tier number (1-5) to the user's message based on task complexity.

Tier definitions:
1 = General & Coding: normal day-to-day work, standard coding, content drafting, Q&A, file operations, status checks, short acknowledgements (default/primary tier)
2 = Enhanced: tasks requiring slightly more reasoning or multi-file context, simple debugging
3 = Analysis: standard troubleshooting, debugging, code reviews, nuanced logical analysis
4 = Architecture: system design, architecture planning, multi-step implementation plans, migration strategy
5 = Deep-think: multi-file refactors, complex debugging, security auditing, algorithmic optimization, high-stakes tasks with many interacting constraints

Rules:
- When unsure between two tiers, pick the LOWER one
- Tier 5 only for truly security-critical or algorithmically dense tasks
- Polish and English messages treated equally
- Consider the INTENT not just keywords

Respond with ONLY a single digit: 1, 2, 3, 4, or 5. Nothing else."""


def _classify_with_flash(user_message: str, conversation_history: list) -> int:
    """Call Flash to classify turn complexity. Returns tier 1-5.

    Attempts the configured primary classifier first (via triage_specifier task),
    then retries each entry in classifier.fallbacks if the primary fails.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.warning("model-router: auxiliary_client unavailable: %s -- defaulting T2", exc)
        return 2

    # Include last 2 assistant turns as context (cheap, small)
    context_turns = []
    assistant_count = 0
    for msg in reversed(conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                context_turns.insert(0, {"role": "assistant", "content": content[:300]})
                assistant_count += 1
                if assistant_count >= 2:
                    break

    messages = [{"role": "system", "content": _CLASSIFIER_SYSTEM}]
    if context_turns:
        messages.append({
            "role": "user",
            "content": "[Recent conversation context]\n"
                       + "\n".join(f"Assistant: {m['content']}" for m in context_turns),
        })
        messages.append({"role": "assistant", "content": "Understood."})
    messages.append({"role": "user", "content": user_message[:800]})

    def _parse_tier(response: Any) -> int | None:
        try:
            raw = response.choices[0].message.content.strip()
            digit = re.search(r"[1-5]", raw)
            return int(digit.group()) if digit else None
        except Exception:
            return None

    # --- Primary attempt (uses triage_specifier task config from config.yaml) ---
    primary_provider = FLASH_PROVIDER or (PROVIDER_PRIORITY[0] if PROVIDER_PRIORITY else "")
    if _is_provider_healthy(primary_provider):
        try:
            response = call_llm(
                task="triage_specifier",
                messages=messages,
                max_tokens=3,
                temperature=0.0,
            )
            tier = _parse_tier(response)
            if tier is not None:
                return tier
        except Exception as exc:
            logger.warning("model-router: classifier primary (%s) failed: %s", primary_provider, exc)
            _mark_provider_failed(primary_provider)
    else:
        logger.info("model-router: classifier primary provider %r unhealthy; skipping to fallbacks", primary_provider)

    # --- Fallback attempts ---
    for fb in CLASSIFIER_FALLBACKS:
        fb_provider = str(fb.get("provider") or "").strip()
        fb_model    = str(fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            continue
        if not _is_provider_healthy(fb_provider):
            logger.debug("model-router: classifier fallback %r unhealthy; skipping", fb_provider)
            continue
        fb_base_url = _PROVIDER_BASE_URLS.get(fb_provider)
        try:
            fb_extra: dict[str, Any] = {}
            fb_reasoning = fb.get("reasoning_effort")
            if fb_reasoning:
                fb_extra["reasoning_effort"] = fb_reasoning
            response = call_llm(
                provider=fb_provider,
                model=fb_model,
                base_url=fb_base_url,
                messages=messages,
                max_tokens=3,
                temperature=0.0,
                extra_body=fb_extra or None,
            )
            tier = _parse_tier(response)
            if tier is not None:
                logger.info("model-router: classifier fallback %r/%s succeeded", fb_provider, fb_model)
                return tier
        except Exception as exc:
            logger.warning("model-router: classifier fallback %r/%s failed: %s", fb_provider, fb_model, exc)
            _mark_provider_failed(fb_provider)

    safe_tier = int(_router_config.get("safe_tier", 2) or 2)
    if safe_tier not in range(1, 6):
        safe_tier = 2
    logger.warning("model-router: all classifier providers exhausted -- defaulting T%d", safe_tier)
    _append_router_event("classifier_fallback", reason="all_providers_exhausted", safe_tier=safe_tier)
    return safe_tier


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

_session_last:    dict[str, tuple[str, int]] = {}  # session_id -> (msg, tier)
_session_manual:  dict[str, str] = {}              # session_id -> model set manually by user
_session_pinned:  dict[str, bool] = {}             # session_id -> True when user used /model or /tN
                                                    # plugin stays OFF until /auto or /new
_last_tier:       dict[str, int] = {}              # session_id -> current assigned tier
_base_tier:       dict[str, int] = {}              # session_id -> tier for this turn (before escalation)
_tool_errors:     dict[str, int] = {}              # session_id -> consecutive tool error count this turn
_escalated:       dict[str, bool] = {}             # session_id -> True if we already escalated mid-turn
_session_working: dict[str, int] = {}              # session_id -> working tier for this turn
_session_floor:   dict[str, int] = {}              # session_id -> floor tier for this turn
_mechanical_streak: dict[str, int] = {}            # session_id -> consecutive read-only tools
_live_agents:     dict[str, Any] = {}              # session_id -> active agent bound by WebUI/CLI bridge
_live_tui_sessions: dict[str, tuple[str, dict]] = {}  # session_id -> (tui_sid, session_dict) for TUI mode
_session_ts:      dict[str, float] = {}            # session_id -> last-seen monotonic timestamp
_SESSION_TTL      = 86_400.0                       # evict sessions idle longer than 24 h
_state_lock = threading.Lock()
_patch_lock = threading.Lock()                     # guards _patch_status_bar double-patch check
_tui_server_module: Any = None                     # cached tui_gateway.server module (pre-loaded at register time)

def get_last_tier(session_id: str) -> int:
    with _state_lock:
        return _last_tier.get(session_id, 0)


def _evict_stale_sessions() -> None:
    """Remove state for sessions idle longer than _SESSION_TTL. Called on each new user turn."""
    cutoff = time.monotonic() - _SESSION_TTL
    with _state_lock:
        stale = [sid for sid, ts in _session_ts.items() if ts < cutoff]
        for sid in stale:
            for d in (
                _session_last, _session_manual, _session_pinned, _last_tier,
                _base_tier, _tool_errors, _escalated, _live_agents, _live_tui_sessions,
                _session_runtime_state, _session_route_trace, _session_ts,
                _session_working, _session_floor, _mechanical_streak,
            ):
                d.pop(sid, None)
    if stale:
        _persist_router_state()
        logger.debug("model-router: evicted %d stale session(s)", len(stale))


# ---------------------------------------------------------------------------
# Provider health tracking
# ---------------------------------------------------------------------------

def _mark_provider_failed(provider: str) -> None:
    """Record a provider failure timestamp for health-check gating."""
    if not provider:
        return
    with _state_lock:
        _provider_failures[provider.lower()] = time.monotonic()
    logger.info("model-router: provider %r marked unhealthy (TTL %ss)", provider, _PROVIDER_UNHEALTHY_TTL)


def _is_provider_healthy(provider: str) -> bool:
    """Return True when provider has not failed recently (or has no recorded failure)."""
    if not provider:
        return True
    with _state_lock:
        failed_at = _provider_failures.get(provider.lower(), 0.0)
    return (time.monotonic() - failed_at) > _PROVIDER_UNHEALTHY_TTL


def _select_tier_entry(tier: int) -> tuple[str, str | None, str]:
    """Back-compat helper returning the resolved runtime tuple for a tier."""
    runtime = resolve_tier_runtime(tier)
    return (
        str(runtime.get("model") or ""),
        runtime.get("reasoning"),
        str(runtime.get("provider") or ""),
    )


def _sync_tier_fallbacks_to_config(tier: int) -> None:
    """Write tier-specific fallback chain to config.yaml fallback_providers.

    This lets hermes' built-in fallback mechanism handle provider-level failures
    during the actual LLM call with the tier-appropriate model chain.
    """
    fallbacks = TIER_FALLBACKS.get(tier, [])
    if not fallbacks:
        return
    try:
        from hermes_cli.config import load_config, save_config  # type: ignore

        cfg = load_config()
        entries = []
        for fb in fallbacks:
            prov  = str(fb.get("provider") or "").strip()
            model = str(fb.get("model") or "").strip()
            if not prov or not model:
                continue
            entry: dict[str, Any] = {"provider": prov, "model": model}
            base_url = _PROVIDER_BASE_URLS.get(prov)
            if base_url:
                entry["base_url"] = base_url
            entries.append(entry)
        if entries:
            cfg["fallback_providers"] = entries
            save_config(cfg)
            logger.debug("model-router: synced T%d fallbacks to config.yaml fallback_providers", tier)
    except Exception as exc:
        logger.debug("model-router: could not sync tier fallbacks to config: %s", exc)


def bind_session_agent(session_id: str, agent: Any) -> None:
    """Bind a live agent instance to a session so WebUI turns are steerable."""
    if not session_id or agent is None:
        return
    with _state_lock:
        _live_agents[session_id] = agent


def unbind_session_agent(session_id: str, agent: Any | None = None) -> None:
    """Remove a previously bound live agent when a WebUI turn ends."""
    if not session_id:
        return
    with _state_lock:
        current = _live_agents.get(session_id)
        if current is None:
            return
        if agent is not None and current is not agent:
            return
        _live_agents.pop(session_id, None)


def is_session_pinned(session_id: str) -> bool:
    """True if auto-routing is disabled for this session (user used /model or /tN)."""
    with _state_lock:
        return _session_pinned.get(session_id, False)


def _get_cached_tier(session_id: str, msg: str) -> int | None:
    with _state_lock:
        entry = _session_last.get(session_id)
        if entry and entry[0] == msg:
            return entry[1]
    return None


def _set_cached_tier(session_id: str, msg: str, tier: int) -> None:
    with _state_lock:
        # Cache one entry per session (msg, tier) for dedup.
        # Keeps only the CURRENT message — new messages will replace old ones.
        # This avoids unbounded growth while preserving dedup for immediate reruns.
        _session_last[session_id] = (msg, tier)


def _is_manual_override(session_id: str, current_model: str) -> bool:
    """True if user manually pinned a model via /model or /tN command.

    Pinned sessions are FULLY blocked — plugin does not touch model or
    reasoning_config until the user explicitly calls /auto (or /new).
    """
    with _state_lock:
        return _session_pinned.get(session_id, False)


def _record_router_set(session_id: str) -> None:
    """Called after the plugin itself applies a tier switch.
    Does NOT clear the pin — only /auto can clear a pin.
    Clears the _session_manual sentinel (used only for auto-detection)."""
    with _state_lock:
        _session_manual.pop(session_id, None)


def _record_route_trace(
    session_id: str,
    reason: str,
    tier: int,
    profile_id: str = "",
    model: str = "",
    provider: str = "",
    route_name: str | None = None,
    route_keyword: str | None = None,
) -> None:
    """Record the routing decision reason for this session."""
    if not session_id:
        return
    trace: dict[str, Any] = {
        "reason": reason,
        "tier": tier,
        "profile_id": profile_id,
        "model": model,
        "provider": provider,
        "updated_at": time.time(),
    }
    if route_name:
        trace["route_name"] = route_name
    if route_keyword:
        trace["route_keyword"] = route_keyword
    with _state_lock:
        _session_route_trace[session_id] = trace


def _record_runtime_state(session_id: str, runtime: dict[str, Any], tier: int) -> None:
    if not session_id:
        return
    state = {
        "tier": tier,
        "profile_id": str(runtime.get("profile_id") or ""),
        "display_name": str(runtime.get("display_name") or ""),
        "model": str(runtime.get("model") or ""),
        "provider": str(runtime.get("provider") or ""),
        "base_url": str(runtime.get("base_url") or ""),
        "api_mode": str(runtime.get("api_mode") or ""),
        "reasoning": runtime.get("reasoning"),
        "updated_at": time.time(),
    }
    with _state_lock:
        _session_runtime_state[session_id] = state
    _persist_router_state()
    _append_router_event(
        "runtime_state_updated",
        session_id,
        tier=tier,
        profile_id=state["profile_id"],
        model=state["model"],
        provider=state["provider"],
        api_mode=state["api_mode"],
        reasoning=state["reasoning"],
    )


def get_session_state(session_id: str) -> dict[str, Any]:
    with _state_lock:
        state = dict(_session_runtime_state.get(session_id, {}))
        state["pinned"] = _session_pinned.get(session_id, False)
        state["last_tier"] = _last_tier.get(session_id, 0)
        state["base_tier"] = _base_tier.get(session_id, 0)
        trace = dict(_session_route_trace.get(session_id, {}))
    if trace:
        state["route_reason"] = trace.get("reason", "")
        state["route_name"] = trace.get("route_name")
        state["route_keyword"] = trace.get("route_keyword")
    return state


def notify_manual_override(session_id: str, model: str) -> None:
    """Called when we auto-detect a /model change mid-session.
    Sets the session pin so we stop routing for this session."""
    with _state_lock:
        _session_manual[session_id] = model
        _session_pinned[session_id] = True
        known_tier = MODEL_TO_TIER.get(str(model or ""), 0)
        if known_tier:
            _last_tier[session_id] = known_tier
            _base_tier[session_id] = known_tier
    _persist_router_state()
    _append_router_event("manual_override_detected", session_id, model=model)


def pin_session(session_id: str, model: str) -> None:
    """Explicit pin — called from /model, /t1-/t5 slash commands.
    Immediately halts auto-routing for this session."""
    with _state_lock:
        _session_manual[session_id] = model
        _session_pinned[session_id] = True
    _persist_router_state()
    _append_router_event("session_pinned", session_id, model=model)
    logger.info("model-router: session %s pinned to %s", session_id, model)


def unpin_session(session_id: str) -> None:
    """Called from /auto slash command. Re-enables auto-routing."""
    with _state_lock:
        _session_manual.pop(session_id, None)
        _session_pinned.pop(session_id, None)
        # Also clear cache so next turn re-classifies fresh
        _session_last.pop(session_id, None)
        _session_runtime_state.pop(session_id, None)
    _persist_router_state()
    _append_router_event("session_unpinned", session_id)
    logger.info("model-router: session %s unpinned, auto-routing resumed", session_id)


# ---------------------------------------------------------------------------
# Status bar injection
# ---------------------------------------------------------------------------

def _patch_status_bar(cli) -> None:
    """Monkey-patch cli._get_status_bar_snapshot to inject [Tx] tier prefix.

    Called once after we have a cli reference. Safe to call multiple times --
    guards against double-patching via _router_patched sentinel (checked under
    _patch_lock to prevent a TOCTOU race on concurrent calls).
    """
    import types
    with _patch_lock:
        if getattr(cli, "_router_patched", False):
            return
        original_snapshot = cli._get_status_bar_snapshot.__func__  # unbound

        def _patched_snapshot(self_cli):
            snap = original_snapshot(self_cli)
            # Read current tier from our registry
            session_id = getattr(getattr(self_cli, "agent", None), "session_id", None) or ""
            tier = get_last_tier(session_id)
            if tier:
                # Read the model name from agent.model directly so the badge
                # and the model name always match (agent.model may have changed
                # since the original snapshot was captured).
                agent = getattr(self_cli, "agent", None)
                current_model = getattr(agent, "model", "") if agent else ""
                if not current_model:
                    current_model = TIERS.get(tier, {}).get("model", "")
                # Extract a short display name, stripping path prefix and
                # Bedrock inference-profile region prefixes.
                name_only = current_model.split("/")[-1] if current_model else "unknown"
                for _bedrock_pfx in ("global.anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic."):
                    if name_only.startswith(_bedrock_pfx):
                        name_only = name_only[len(_bedrock_pfx):]
                        break
                # Read provider from runtime state; fall back to the tier's *primary*
                # configured provider for new sessions (avoids showing a fallback provider).
                _provider_raw = ""
                with _state_lock:
                    _provider_raw = str(_session_runtime_state.get(session_id, {}).get("provider") or "")
                if not _provider_raw:
                    try:
                        _pid = _tier_profile_id(tier)
                        _primary = RUNTIME_PROFILES.get(_pid, {})
                        _provider_raw = str(
                            _primary.get("provider")
                            or TIERS.get(tier, {}).get("provider")
                            or ""
                        )
                    except Exception:
                        pass
                provider_short = _PROVIDER_SHORT.get(_provider_raw.lower(), _provider_raw[:6]) if _provider_raw else ""
                if provider_short:
                    snap["model_short"] = f"[{provider_short}|T{tier}] {name_only}"
                else:
                    snap["model_short"] = f"[T{tier}] {name_only}"
            return snap

        cli._get_status_bar_snapshot = types.MethodType(_patched_snapshot, cli)
        cli._router_patched = True
        logger.debug("model-router: status bar patch applied")


# ---------------------------------------------------------------------------
# Live agent / routing helpers
# ---------------------------------------------------------------------------

def _get_live_agent(session_id: str = "") -> Any | None:
    """Return the active agent for this session from CLI, gateway, or TUI bindings."""
    with _state_lock:
        bound_agent = _live_agents.get(session_id) if session_id else None
    if bound_agent is not None:
        return bound_agent

    if _manager_ref is None:
        return None

    # CLI mode: direct agent reference via _cli_ref
    try:
        cli = _manager_ref._cli_ref
        agent = getattr(cli, "agent", None) if cli else None
        if agent is not None:
            if not session_id:
                return agent
            agent_session = getattr(agent, "session_id", "") or ""
            if agent_session == session_id:
                return agent
    except Exception:
        pass

    # TUI gateway mode: scan tui_gateway.server._sessions for matching agent.
    # _cli_ref is None in TUI mode; the desktop TUI uses its own server module
    # with a _sessions dict keyed by TUI client session IDs.
    # _tui_server_module is pre-cached at register() time to avoid per-call import races.
    if session_id and _tui_server_module is not None:
        try:
            _tui_sessions: dict = getattr(_tui_server_module, "_sessions", {})
            _tui_lock = getattr(_tui_server_module, "_sessions_lock", None)
            ctx = _tui_lock if _tui_lock is not None else contextlib.nullcontext()
            found_agent = None
            found_sid: str = ""
            found_sess: dict = {}
            with ctx:
                for _sid_key, _sess in _tui_sessions.items():
                    _a = _sess.get("agent")
                    if _a is None:
                        continue
                    # session_key is kept in sync with agent.session_id (even after
                    # compression rotations update both fields together).
                    if _sess.get("session_key") == session_id or getattr(_a, "session_id", "") == session_id:
                        found_agent = _a
                        found_sid = _sid_key
                        found_sess = _sess
                        break
            if found_agent is not None:
                with _state_lock:
                    _live_agents[session_id] = found_agent
                    _live_tui_sessions[session_id] = (found_sid, found_sess)
                return found_agent
        except Exception as exc:
            logger.debug("model-router: TUI scan error: %s", exc)

    return None


def _target_tier_for_turn(session_id: str, msg: str, history: list, current_model: str) -> tuple[int, bool]:
    """Return (target_tier, is_new_user_turn) for the current message."""
    with _state_lock:
        last_entry = _session_last.get(session_id)
        is_new_user_turn = (last_entry is None) or (last_entry[0] != msg)

    if is_new_user_turn:
        _evict_stale_sessions()
        with _state_lock:
            _tool_errors[session_id] = 0
            _escalated[session_id] = False
            _session_ts[session_id] = time.monotonic()
            _mechanical_streak[session_id] = 0

        route_match: dict[str, Any] | None = None
        route_reason = "classifier"

        explicit = _detect_explicit_tier(msg)
        if explicit is not None:
            target_tier = explicit
            working = target_tier
            floor = target_tier
            route_reason = "explicit_tier"
            _set_cached_tier(session_id, msg, target_tier)
            logger.info("model-router: explicit T%d request honoured", target_tier)
        elif _is_obvious_ack(msg):
            target_tier = 1
            working = target_tier
            floor = target_tier
            route_reason = "ack"
            _set_cached_tier(session_id, msg, target_tier)
        else:
            route_match = _match_task_route(msg)
            if route_match is not None:
                target_tier = route_match["tier"]
                working = route_match["working_tier"]
                floor = route_match["floor_tier"]
                route_reason = "task_route"
                _set_cached_tier(session_id, msg, target_tier)
                logger.info(
                    "model-router: task route %r matched keyword %r -> T%d [working T%d, floor T%d]",
                    route_match["name"], route_match["keyword"], target_tier, working, floor,
                )
            else:
                target_tier = _classify_with_flash(msg, history)
                working = target_tier
                delta = int(_router_config.get("default_floor_delta", 2) or 2)
                floor = max(1, target_tier - delta)
                _set_cached_tier(session_id, msg, target_tier)

        with _state_lock:
            _base_tier[session_id] = target_tier
            _last_tier[session_id] = target_tier
            _session_working[session_id] = working
            _session_floor[session_id] = floor

        _rt = resolve_tier_runtime(target_tier)
        _rt_profile_id = str(_rt.get("profile_id") or "")
        _rt_model = str(_rt.get("model") or "")
        _rt_provider = str(_rt.get("provider") or "")
        _record_route_trace(
            session_id, route_reason, target_tier,
            profile_id=_rt_profile_id, model=_rt_model, provider=_rt_provider,
            route_name=route_match["name"] if route_match else None,
            route_keyword=route_match["keyword"] if route_match else None,
        )
        _append_router_event(
            "route_decision",
            session_id,
            reason=route_reason,
            tier=target_tier,
            profile_id=_rt_profile_id,
            model=_rt_model,
            provider=_rt_provider,
            route_name=route_match["name"] if route_match else None,
            route_keyword=route_match["keyword"] if route_match else None,
        )
        return target_tier, True

    with _state_lock:
        _session_ts[session_id] = time.monotonic()
        last_tier = _last_tier.get(session_id, 2)
        if session_id not in _session_working:
            _session_working[session_id] = last_tier
        if session_id not in _session_floor:
            delta = int(_router_config.get("default_floor_delta", 2) or 2)
            _session_floor[session_id] = max(1, last_tier - delta)
        return last_tier, False


def prepare_turn(
    *,
    session_id: str,
    user_message: str,
    conversation_history: list | None = None,
    current_model: str = "",
    platform: str = "",
    apply_live: bool = False,
) -> dict[str, Any]:
    """Classify the turn and optionally apply the routed tier to a live agent."""
    msg = user_message.strip()
    if not msg:
        return {
            "session_id": session_id,
            "pinned": is_session_pinned(session_id),
            "tier": get_last_tier(session_id) or MODEL_TO_TIER.get(current_model, 0),
            "model": current_model,
            "reasoning": None,
            "platform": platform,
            "is_new_turn": False,
        }

    history = conversation_history or []
    actual_model = current_model
    agent = _get_live_agent(session_id)
    if agent is not None:
        actual_model = getattr(agent, "model", actual_model) or actual_model

    if _is_manual_override(session_id, actual_model):
        state = get_session_state(session_id)
        logger.debug("model-router: manual override active (%s), skipping", actual_model)
        return {
            "session_id": session_id,
            "pinned": True,
            "tier": state.get("tier") or MODEL_TO_TIER.get(actual_model, get_last_tier(session_id)),
            "model": actual_model,
            "reasoning": state.get("reasoning")
            or (getattr(agent, "reasoning_config", None) if agent is not None else None),
            "platform": platform,
            "is_new_turn": False,
        }

    target_tier, is_new_user_turn = _target_tier_for_turn(session_id, msg, history, actual_model)
    target_runtime = resolve_tier_runtime(target_tier)
    target_model = str(target_runtime.get("model") or "")
    target_reasoning = target_runtime.get("reasoning")
    target_provider = str(target_runtime.get("provider") or "")
    if not target_model:
        target_model = actual_model
    actual_provider = getattr(agent, "provider", "") if agent is not None else ""
    provider_mismatch = bool(target_provider and actual_provider and target_provider != actual_provider)

    logger.debug(
        "model-router: turn T%d -> provider=%s model=%s vs actual_provider=%s actual_model=%s",
        target_tier, target_provider, target_model, actual_provider, actual_model,
    )

    if apply_live:
        if target_model != actual_model or provider_mismatch:
            logger.info(
                "model-router: switching T%d (was T%d / %s via %s)",
                target_tier,
                MODEL_TO_TIER.get(actual_model, 0),
                actual_model.split("/")[-1] if actual_model else "unknown",
                actual_provider or "unknown",
            )
            _apply_tier(session_id, target_tier, actual_model, source="")
        else:
            if agent is not None:
                _sync_fallback_chain(agent, target_tier)
            try:
                cli = _manager_ref._cli_ref if _manager_ref is not None else None
                if cli:
                    _patch_status_bar(cli)
            except Exception:
                pass
    elif agent is not None and (target_model != actual_model or provider_mismatch):
        # WebUI may pre-resolve the routed model before the hook fires; if a
        # reused cached agent still carries the old model/provider, normalize it here.
        _apply_tier(session_id, target_tier, actual_model, source="webui-sync")
    elif agent is not None:
        _sync_fallback_chain(agent, target_tier)

    return {
        "session_id": session_id,
        "pinned": False,
        "tier": target_tier,
        "model": target_model,
        "reasoning": target_reasoning,
        "platform": platform,
        "is_new_turn": is_new_user_turn,
    }


# ---------------------------------------------------------------------------
# Apply model switch helper
# ---------------------------------------------------------------------------

def _sync_fallback_chain(agent: Any, target_tier: int) -> None:
    tier_fb = TIER_FALLBACKS.get(target_tier, [])
    if tier_fb:
        entries = []
        for fb in tier_fb:
            prov = str(fb.get("provider") or "").strip()
            fb_model = str(fb.get("model") or "").strip()
            if not prov or not fb_model:
                continue
            entry: dict[str, Any] = {"provider": prov, "model": fb_model}
            fb_base_url = _PROVIDER_BASE_URLS.get(prov)
            if fb_base_url:
                entry["base_url"] = fb_base_url
            # preserve reasoning if present
            if "reasoning" in fb:
                entry["reasoning"] = fb["reasoning"]
            elif "reasoning_effort" in fb:
                entry["reasoning_effort"] = fb["reasoning_effort"]
            entries.append(entry)
        
        if entries:
            agent._fallback_chain = entries
            agent._fallback_index = 0
            agent._fallback_model = entries[0]
            agent._fallback_activated = False


def _apply_tier(session_id: str, target_tier: int, current_model: str, source: str = "") -> None:
    """Switch agent.model + reasoning_config to target_tier. Emits badge.

    Uses provider-correct runtime profile resolution for the selected tier.
    """
    global _manager_ref

    if _manager_ref is None:
        return

    runtime = resolve_tier_runtime(target_tier)
    target_model = str(runtime.get("model") or "")
    target_reasoning = runtime.get("reasoning")
    provider = str(runtime.get("provider") or "")
    current_tier = MODEL_TO_TIER.get(current_model, 2)

    # Apply status bar patch lazily (needs cli ref)
    try:
        cli = _manager_ref._cli_ref
        if cli:
            _patch_status_bar(cli)
    except Exception:
        pass

    try:
        cli = _manager_ref._cli_ref
        agent = _get_live_agent(session_id)
        if agent is None:
            _record_runtime_state(session_id, runtime, target_tier)
            return

        old_model = agent.model
        old_provider = getattr(agent, "provider", "") or ""
        base_url = str(runtime.get("base_url") or _PROVIDER_BASE_URLS.get(provider, "")).strip()
        api_mode = str(runtime.get("api_mode") or _determine_api_mode(provider, base_url)).strip()

        # In TUI mode, invoke the proper switch mechanism so the shared client,
        # provider, base_url, api_mode, and status bar all update via the same
        # path as /model. Direct attribute assignment is not enough there.
        tui_ctx = None
        with _state_lock:
            tui_ctx = _live_tui_sessions.get(session_id)
        if tui_ctx is not None and _tui_server_module is not None:
            try:
                _sid_tui, _sess_tui = tui_ctx
                model_spec = f"{target_model} --provider {provider}"
                _tui_server_module._apply_model_switch(
                    _sid_tui, _sess_tui, model_spec,
                    confirm_expensive_model=False,
                    pin_session_override=False,
                )
            except Exception as exc:
                logger.debug("model-router: TUI _apply_model_switch failed: %s", exc)
                raise
        elif hasattr(agent, "switch_model"):
            agent.switch_model(
                new_model=target_model,
                new_provider=provider,
                api_key="",
                base_url=base_url,
                api_mode=api_mode,
            )
        else:
            agent.model = target_model
            if provider:
                agent.provider = provider
            agent.base_url = base_url or None
            if api_mode:
                agent.api_mode = api_mode

        if target_reasoning:
            normalized_reasoning = _normalize_reasoning_for_provider(target_reasoning, provider)
            agent.reasoning_config = {"effort": normalized_reasoning} if normalized_reasoning else None
        else:
            agent.reasoning_config = None

        _record_router_set(session_id)
        _record_runtime_state(session_id, runtime, target_tier)

        with _state_lock:
            _last_tier[session_id] = target_tier

        # Set the agent's fallback chain directly for this tier instead of syncing to config.yaml
        _sync_fallback_chain(agent, target_tier)

        emoji, label = _TIER_LABELS.get(target_tier, ("", f"T{target_tier}"))
        src_tag = f" [{source}]" if source else ""
        tier_msg = f"{emoji} model-router: {label} ({target_model.split('/')[-1]}){src_tag}"

        _vprint = getattr(cli, "_vprint", None) or getattr(cli, "_cprint", None)
        if _vprint:
            try:
                _vprint(tier_msg)
            except Exception:
                logger.info(tier_msg)
        else:
            logger.info(tier_msg)

        logger.info(
            "model-router: T%d->T%d | %s -> %s%s",
            current_tier, target_tier,
            f"{old_provider}/{old_model}", f"{provider}/{target_model}",
            f" [{source}]" if source else "",
        )
    except Exception as exc:
        logger.warning("model-router: failed to apply switch: %s", exc)


# ---------------------------------------------------------------------------
# Hook: pre_llm_call  (fires before EVERY LLM call in the loop)
# ---------------------------------------------------------------------------

def on_pre_llm_call(
    *,
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = True,
    model: str = "",
    session_id: str = "",
    platform: str = "",
    **kwargs: Any,
) -> None:
    if _manager_ref is None:
        return

    prepare_turn(
        session_id=session_id,
        user_message=user_message,
        conversation_history=conversation_history,
        current_model=model,
        platform=platform,
        apply_live=True,
    )


# ---------------------------------------------------------------------------
# Hook: post_tool_call  (fires after every tool execution inside the loop)
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS = {
    "read_file", "view_file", "list_dir", "grep_search", "search_files",
    "web_search", "web_extract", "x_search", "session_search",
    "read_terminal", "skills_list", "skill_view", "vision_analyze",
    "video_analyze", "memory", "read_resource", "list_resources",
    "read_url_content", "list_permissions"
}

_MECHANICAL_STREAK_THRESHOLD = 1

# How many consecutive tool errors trigger a self-escalation
_ESCALATION_ERROR_THRESHOLD = 2

def on_post_tool_call(
    *,
    tool_name: str = "",
    result: str | None = None,
    session_id: str = "",
    **kwargs: Any,
) -> None:
    """Self-escalation: if the agent keeps hitting tool errors, bump one tier.
    Also implements per-call routing heuristics: read-only sequences drop to
    the task floor; errors or write/exec tools restore the working tier.

    Only escalates ONCE per turn (guards against ping-pong).
    Only escalates if currently below T4 (T4/T5 are already the strongest).
    Resets error counter on success.
    Does NOT escalate/downgrade if session is pinned by user via /model or /tN.
    """
    if _manager_ref is None or not session_id:
        return

    # Respect session pin — never auto-escalate or downgrade a pinned session
    with _state_lock:
        if _session_pinned.get(session_id, False):
            return

    # Detect error in result
    is_error = False
    if result is not None:
        result_lower = result[:500].lower()
        if (
            '"error"' in result_lower
            or '"failed"' in result_lower
            or result_lower.startswith("error")
            or (
                "exit_code" in result_lower
                and '"exit_code": ' in result_lower
                and '"exit_code": 0' not in result_lower
                and '"exit_code": null' not in result_lower
            )
        ):
            is_error = True

    with _state_lock:
        if is_error:
            _tool_errors[session_id] = _tool_errors.get(session_id, 0) + 1
        else:
            _tool_errors[session_id] = 0  # reset on success

        error_count = _tool_errors.get(session_id, 0)
        current_tier = _last_tier.get(session_id, 2)

    if (
        is_error
        and error_count >= _ESCALATION_ERROR_THRESHOLD
        and current_tier < 4
    ):
        new_tier = min(current_tier + 1, 4)
        with _state_lock:
            _escalated[session_id] = True
            _last_tier[session_id] = new_tier
            _tool_errors[session_id] = 0  # Reset counter so next 2 errors can trigger another escalation

        agent = _get_live_agent(session_id)
        current_model = getattr(agent, "model", TIERS[current_tier]["model"]) if agent else TIERS[current_tier]["model"]

        logger.info(
            "model-router: self-escalating T%d->T%d after %d tool errors",
            current_tier, new_tier, error_count,
        )
        _apply_tier(session_id, new_tier, current_model, source="auto-escalate")

    # Step 3 — Per-call heuristic downgrade/restore logic
    is_read_only = bool(tool_name and tool_name in _READ_ONLY_TOOLS)

    with _state_lock:
        working = _session_working.get(session_id)
        floor = _session_floor.get(session_id)
        # Fallback initialization if missing
        if working is None or floor is None:
            base = _base_tier.get(session_id, 2)
            if working is None:
                working = base
                _session_working[session_id] = working
            if floor is None:
                delta = int(_router_config.get("default_floor_delta", 2) or 2)
                floor = max(1, working - delta)
                _session_floor[session_id] = floor

        # Fetch current tier again (in case it was escalated above)
        current_tier = _last_tier.get(session_id, 2)

    if not is_read_only or is_error:
        with _state_lock:
            _mechanical_streak[session_id] = 0
        if current_tier < working:
            with _state_lock:
                _last_tier[session_id] = working
            agent = _get_live_agent(session_id)
            current_model = getattr(agent, "model", TIERS[current_tier]["model"]) if agent else TIERS[current_tier]["model"]
            logger.info(
                "model-router: restoring working tier T%d (was T%d) due to %s",
                working, current_tier, "tool error" if is_error else f"non-read-only tool {tool_name}"
            )
            _apply_tier(session_id, working, current_model, source="restore-working")
    else:
        with _state_lock:
            streak = _mechanical_streak.get(session_id, 0) + 1
            _mechanical_streak[session_id] = streak
        if streak >= _MECHANICAL_STREAK_THRESHOLD and current_tier > floor:
            with _state_lock:
                _last_tier[session_id] = floor
            agent = _get_live_agent(session_id)
            current_model = getattr(agent, "model", TIERS[current_tier]["model"]) if agent else TIERS[current_tier]["model"]
            logger.info(
                "model-router: downgrading to floor tier T%d (was T%d) after %d read-only tool(s)",
                floor, current_tier, streak
            )
            _apply_tier(session_id, floor, current_model, source="mechanical-floor")


# ---------------------------------------------------------------------------
# Hook: post_llm_call  (fires after every LLM response in the loop)
# ---------------------------------------------------------------------------

def on_post_llm_call(
    *,
    model: str = "",
    session_id: str = "",
    **kwargs: Any,
) -> None:
    """Two responsibilities:

    1. Detect user manual /model change and pin the session (stop auto-routing).
    2. De-escalate: if we escalated mid-turn, restore base tier now that
       the heavy work is done (fires after the final response is complete).
       Only de-escalates if is_first_turn would be True next call, i.e.
       we detect this is the FINAL response (no pending tool calls).
       We approximate this by checking if the current call was NOT mid-loop.
    """
    global _manager_ref
    if _manager_ref is None:
        return

    try:
        agent = _get_live_agent(session_id)
        if agent is None:
            return

        # 1. Detect external manual model change (e.g. user ran /model mid-session)
        #    agent.model was changed outside of our _apply_tier call.
        #    We detect this by comparing agent.model to what we last set.
        with _state_lock:
            already_pinned = _session_pinned.get(session_id, False)
            router_last_manual = _session_manual.get(session_id)

        if not already_pinned:
            # Check if agent.model is now something we didn't set
            with _state_lock:
                last_router_tier = _last_tier.get(session_id, 0)
            expected_model = TIERS[last_router_tier]["model"] if last_router_tier else None

            if expected_model and agent.model != expected_model and agent.model != model:
                # Model changed between our last set and now — user did /model
                notify_manual_override(session_id, agent.model)
                logger.info(
                    "model-router: detected manual model change to %s -- pinning session, auto-routing paused",
                    agent.model,
                )
                return

        # Sync status bar if _try_activate_fallback() changed the model mid-turn.
        # _session_runtime_state is only updated by _apply_tier(); when the hermes
        # fallback chain fires during an API call our recorded model/provider drifts
        # from agent.model. Fix both the CLI provider badge and the TUI session.info
        # display by snapshotting the actual runtime here (fires after every LLM
        # response, so the lag is at most one round-trip).
        if getattr(agent, "_fallback_activated", False):
            _fb_model = getattr(agent, "model", "")
            _fb_provider = getattr(agent, "provider", "")
            with _state_lock:
                _fb_recorded = dict(_session_runtime_state.get(session_id, {}))
            if _fb_model and (
                _fb_model != _fb_recorded.get("model")
                or _fb_provider != _fb_recorded.get("provider")
            ):
                _fb_recorded["model"] = _fb_model
                _fb_recorded["provider"] = _fb_provider
                with _state_lock:
                    _session_runtime_state[session_id] = _fb_recorded
                with _state_lock:
                    _fb_tui_ctx = _live_tui_sessions.get(session_id)
                if _fb_tui_ctx is not None and _tui_server_module is not None:
                    try:
                        _fb_tui_sid, _fb_tui_sess = _fb_tui_ctx
                        _tui_server_module._emit(
                            "session.info", _fb_tui_sid,
                            _tui_server_module._session_info(agent, _fb_tui_sess),
                        )
                    except Exception as _fb_exc:
                        logger.debug("model-router: fallback status sync failed: %s", _fb_exc)

        # If already pinned, do nothing (no de-escalation either)
        if already_pinned:
            return

        # 2. De-escalate after heavy work completes
        with _state_lock:
            was_escalated = _escalated.get(session_id, False)
            base = _base_tier.get(session_id, 2)
            current = _last_tier.get(session_id, 2)

        if (was_escalated and current > base) or (current < base):
            logger.info(
                "model-router: restoring base/working tier T%d (was T%d) after turn completed",
                base, current,
            )
            with _state_lock:
                _escalated[session_id] = False
                _last_tier[session_id] = base
            _apply_tier(session_id, base, agent.model, source="de-escalate")

    except Exception as exc:
        logger.debug("model-router: on_post_llm_call hook failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Hook: pre_gateway_dispatch  (fires before every incoming message is dispatched)
# ---------------------------------------------------------------------------

def on_pre_gateway_dispatch(*, gateway=None, **kwargs: Any) -> None:
    """Populate _live_agents from the gateway's running session pool.

    In TUI/gateway mode _manager_ref._cli_ref is None, so _get_live_agent falls
    back to _live_agents.  We refresh this cache on every dispatch so each
    agent's session_id → agent mapping is current when pre_llm_call fires.
    """
    if gateway is None:
        return
    try:
        running_agents = getattr(gateway, "_running_agents", {})
        with _state_lock:
            for agent in running_agents.values():
                if agent is None:
                    continue
                sid = getattr(agent, "session_id", None) or ""
                if sid and hasattr(agent, "model"):
                    _live_agents[sid] = agent
    except Exception as exc:
        logger.debug("model-router: pre_gateway_dispatch agent sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Hook: api_request_error  (fires on every failed LLM API call in the loop)
# ---------------------------------------------------------------------------

def on_api_request_error(
    *,
    provider: str = "",
    session_id: str = "",
    error: dict | None = None,
    retryable: bool | None = None,
    **kwargs: Any,
) -> None:
    """Track provider failures for proactive health-gating.

    Only marks a provider unhealthy on non-retryable errors to avoid
    penalizing transient blips that hermes already retries automatically.
    """
    if not provider:
        return
    # Skip retryable transient errors (5xx, 408, partial reads) — hermes
    # will retry those automatically and we shouldn't penalize the provider.
    if retryable is True:
        return
    _mark_provider_failed(provider)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    global _manager_ref
    _load_router_config()
    _load_persisted_state()
    _manager_ref = ctx._manager
    ctx.register_hook("pre_llm_call",          on_pre_llm_call)
    ctx.register_hook("post_llm_call",         on_post_llm_call)
    ctx.register_hook("post_tool_call",        on_post_tool_call)
    ctx.register_hook("api_request_error",     on_api_request_error)
    ctx.register_hook("pre_gateway_dispatch",  on_pre_gateway_dispatch)

    # Expose public API on the PluginManager so slash commands (/t1-/t5, /auto)
    # can call us via get_plugin_manager().router_apply_tier(...).
    # NOTE: ctx is a PluginContext facade, but cli.py reads from the manager
    # directly via get_plugin_manager(). Setting on ctx._manager ensures the
    # attributes are visible where the handlers look for them.
    mgr = ctx._manager
    mgr.router_pin_session   = pin_session
    mgr.router_unpin_session = unpin_session
    mgr.router_apply_tier    = _apply_tier_by_num
    mgr.router_resolve_tier_runtime = resolve_tier_runtime
    mgr.router_is_pinned     = is_session_pinned
    mgr.router_get_tier      = get_last_tier
    mgr.router_get_tier_meta = get_tier_meta
    mgr.router_get_session_state = get_session_state
    mgr.router_get_recent_events = get_recent_events
    mgr.router_get_diagnostics = get_router_diagnostics
    mgr.router_get_startup_status = get_router_startup_status
    mgr.router_get_analytics = get_router_analytics
    mgr.router_eval_routing = eval_task_routing
    mgr.router_reload_config = _load_router_config
    mgr.router_prepare_turn  = prepare_turn
    mgr.router_bind_agent    = bind_session_agent
    mgr.router_unbind_agent  = unbind_session_agent

    # Pre-cache tui_gateway.server at startup so the module is guaranteed to be
    # importable when _get_live_agent fires during a TUI session. Importing here
    # (vs. per-call) means failure shows up in the log at load time, not silently
    # per-turn inside an except-pass block.
    global _tui_server_module
    try:
        import tui_gateway.server as _tui_srv_mod  # noqa: PLC0415
        _tui_server_module = _tui_srv_mod
        logger.debug("model-router: tui_gateway.server cached at register time")
    except ImportError:
        logger.debug("model-router: tui_gateway.server not available (non-TUI mode)")

    logger.info(
        "model-router: registered -- Flash T1-T5 routing | explicit hints | "
        "self-escalation | de-escalation | status bar [Tx] | /tN + /auto commands"
    )


def get_tier_meta(tier_num: int) -> dict[str, Any]:
    """Return model metadata for a tier so the CLI does not hardcode slugs."""
    if tier_num not in TIERS:
        return {}
    meta = dict(_router_config["tiers"][tier_num])
    runtime = resolve_tier_runtime(tier_num)
    if runtime:
        meta.setdefault("target", runtime.get("profile_id"))
        meta["provider"] = runtime.get("provider") or meta.get("provider")
        meta["base_url"] = runtime.get("base_url") or meta.get("base_url", "")
        meta["api_mode"] = runtime.get("api_mode") or meta.get("api_mode", "")
        meta["display_name"] = runtime.get("display_name") or meta.get("label")
    return meta


def _apply_tier_by_num(session_id: str, tier_num: int, current_model: str) -> dict[str, Any]:
    """Public entry point for /t1-/t5 slash commands.

    Sets the tier, applies the model switch, and pins the session so
    auto-routing does not override the choice.
    """
    if tier_num not in TIERS:
        logger.warning("model-router: invalid tier %d requested", tier_num)
        return {}
    runtime = resolve_tier_runtime(tier_num)
    target_model = str(runtime.get("model") or TIERS[tier_num]["model"])
    pin_session(session_id, target_model)
    with _state_lock:
        _last_tier[session_id] = tier_num
        _base_tier[session_id] = tier_num
    _apply_tier(session_id, tier_num, current_model, source="user-pin")
    state = get_session_state(session_id)
    if state:
        runtime = dict(runtime)
        runtime["tier"] = state.get("tier", tier_num)
    return runtime
