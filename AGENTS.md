# AGENTS.md — model-router developer orientation

Quick orientation for AI agents working on this codebase. Covers
architecture invariants, key functions, and gotchas discovered through
debugging.

---

## Source of truth

| What | Where |
|---|---|
| Runtime routing config | `~/.hermes/model_router.yaml` (active profile) |
| Fallback defaults (bootstrap only) | `DEFAULT_ROUTER_CONFIG` dict in `__init__.py` |
| Bootstrap install | `install.py` / `install.sh` |
| Plugin manifest | `plugin.yaml` |

The plugin loads `model_router.yaml` at `register()` time and on every
`_load_router_config()` call. `DEFAULT_ROUTER_CONFIG` in `__init__.py` is
only used when the yaml file does not exist or is invalid — it is **not**
the active config at runtime.

---

## Five-tier contract

Exactly five tier slots (1–5) must exist at all times. `/t1`–`/t5` always
map to those slots. You can change the model, label, emoji, and reasoning
mode per slot; you must not add or remove slots.

---

## Two separate execution paths

The plugin serves two runtimes that share `pre_llm_call` but have
completely separate session management:

### CLI path (`hermes` terminal client)
- Agent is accessible via `_manager_ref._cli_ref.agent`
- `_get_live_agent()` finds it through `_manager_ref._cli_ref`
- Changing `agent.model` is sufficient to switch models

### TUI path (`tui_gateway/server.py` desktop UI)
- `_cli_ref` is **None** — the TUI does not use the CLI manager ref
- Sessions live in `tui_gateway.server._sessions` (module-level dict,
  keyed by TUI transport sid)
- Each session dict has `"agent"` key and `"session_key"` field (= `agent.session_id`)
- **Direct `agent.model = ...` assignment is not enough in TUI mode** — the
  TUI's OpenAI client is created `shared=True` at agent init; provider,
  base_url, api_key, and api_mode must all be updated together via
  `agent.switch_model()`
- The correct TUI switch path is `tui_gateway.server._apply_model_switch(sid,
  session_dict, "model_id --provider provider", confirm_expensive_model=False,
  pin_session_override=False)` — this calls `agent.switch_model()` and emits
  `session.info` so the status bar updates

### `pre_gateway_dispatch` only fires for the messaging gateway
The `on_pre_gateway_dispatch` hook fires from `gateway/run.py` only (Telegram
/ WhatsApp deployments). It does **not** fire for the desktop TUI. The TUI
is a separate process using `tui_gateway/server.py`.

---

## Key globals in `__init__.py`

| Global | Purpose |
|---|---|
| `_live_agents` | `session_id → agent` — populated from CLI ref or TUI scan |
| `_live_tui_sessions` | `session_id → (tui_sid, session_dict)` — cached when TUI scan succeeds; needed to call `_apply_model_switch` |
| `_tui_server_module` | Cached `tui_gateway.server` module, imported once at `register()` time to avoid per-call import races |
| `_manager_ref` | Reference to the Hermes PluginManager, set in `register()` |
| `_state_lock` | `threading.Lock` guarding all of the above dicts |
| `TIERS` | `{1..5: {model, reasoning, label, emoji, role, best_for}}` — built from the active yaml |
| `MODEL_TO_TIER` | Reverse map, `model_string → tier_int` |
| `PROVIDER_PRIORITY` | Ordered list of providers from yaml (default: `[nous, openai-codex, openrouter]`) |
| `TIER_FALLBACKS` | `{tier_int: [{provider, model, reasoning}, ...]}` |
| `_provider_failures` | `{provider → monotonic timestamp}` — health tracking; TTL 120 s |

---

## Key functions

### `_get_live_agent(session_id)` → agent | None
Finds the active agent for a session in three steps:
1. Check `_live_agents` cache
2. CLI mode: read `_manager_ref._cli_ref.agent`
3. TUI mode: scan `_tui_server_module._sessions` for matching `session_key`
   or `agent.session_id`; on hit, populates both `_live_agents` and
   `_live_tui_sessions`

### `_apply_tier(session_id, target_tier, current_model, source="")`
Switches the agent to `target_tier`. Sequence:
1. `_select_tier_entry(tier)` → `(model, reasoning, provider)`, respecting
   provider health
2. `_get_live_agent(session_id)` — returns early if None
3. `agent.model = target_model` + `agent.reasoning_config = ...`
4. If `_live_tui_sessions[session_id]` exists, call
   `_tui_server_module._apply_model_switch(...)` so the full TUI switch runs
5. Log tier transition

### `_select_tier_entry(tier)` → `(model, reasoning, provider)`
Picks the primary provider if healthy, otherwise walks `TIER_FALLBACKS`
until it finds a healthy one. Returns primary if all are unhealthy (best-
effort).

### `register(ctx)`
Called once at plugin load. Sets `_manager_ref`, registers all hooks,
exposes public API on the PluginManager (`router_pin_session`,
`router_apply_tier`, etc.), and imports `tui_gateway.server` into
`_tui_server_module`.

### `prepare_turn(session_id, user_message, ...)`
Full turn-routing logic: classify message, detect explicit tier hints,
handle obvious acks, check session pin, call `_apply_tier`. Entry point
for `on_pre_llm_call`.

---

## Session eviction

`_evict_stale_sessions()` runs on each new user turn and removes state for
sessions idle > 24 h. It clears `_session_last`, `_live_agents`,
`_live_tui_sessions`, and all other per-session dicts atomically under
`_state_lock`.

---

## Reasoning config

`reasoning` field in `model_router.yaml` is a string parsed by
`hermes_constants.parse_reasoning_effort()`:
- `"medium"` → `{"enabled": True, "effort": "medium"}`
- `"high"` / `"max"` → similar
- `None` / absent → no reasoning

The Nous provider wraps this as `extra_body["reasoning"] = {...}`.

---

## What install.sh does (not the plugin)

`install.sh` / `install.py` patch Hermes core files (`commands.py`,
`cli.py`), sync `auxiliary.triage_specifier` in `config.yaml`, and
generate `skill_routing.md` + a managed block in `SOUL.md`. Those generated
files live in `~/.hermes/`, not in this repo. Re-running install is safe
and idempotent.
