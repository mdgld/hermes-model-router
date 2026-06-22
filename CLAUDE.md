# model-router Plugin - Development Guidelines

## Structure
- `__init__.py` — runtime plugin loaded by Hermes at session start
- `install.py` — standalone installer; **intentionally has no imports from `__init__.py`** (self-contained by design)

## Key Invariants
- `DEFAULT_ROUTER_CONFIG` is duplicated in both files — they must stay in sync (marked with `# KEEP IN SYNC` comments)
- Config files written by installer should always get `chmod(0o600)` — may contain API keys
- Session state dicts (`_session_last`, `_session_manual`, etc.) share a single `_state_lock`; always acquire it before reading or writing
- Models that only support on/off reasoning (e.g. `xiaomi/mimo`, `minimax/minimax`): use `"enabled"` not a level string (`"high"`, `"max"`, etc.)

## Syntax Check
Use the hermes venv:
`/Users/matthewgold/.hermes/hermes-agent/.venv/bin/python3 -m py_compile __init__.py install.py`

## YAML Verification
Use the hermes venv — system `python3` lacks PyYAML:
`/Users/matthewgold/.hermes/hermes-agent/.venv/bin/python3 -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('/Users/matthewgold/.hermes/model_router.yaml').read_text()); print('OK')"`

## Bedrock Routing
- `agent.client = None` is correct for bedrock — transport is `BedrockTransport` (boto3), no OpenAI SDK client
- `determine_api_mode("bedrock", "")` → `"bedrock_converse"` (see `hermes_cli/providers.py`)
- `switch_model()` in `hermes-agent/agent/agent_runtime_helpers.py` must have `elif api_mode == "bedrock_converse":` branch or it falls to the OpenAI path and rolls back every switch
- Bedrock thinking: `additionalModelRequestFields = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}` — `effort` must be in `output_config`, NOT inside `thinking` (causes ValidationException)
- Opus 4.8 on Bedrock: only `type: "adaptive"` works — `"enabled"` + `budget_tokens` → 400. Omit `additionalModelRequestFields` entirely to disable thinking.
- Sonnet 4.6: adaptive recommended; `"enabled"` + `budget_tokens` deprecated but functional
- `build_converse_kwargs` in `bedrock_adapter.py` translates `reasoning_config={"effort": level}` → correct Converse payload; `BedrockTransport` now passes it through (was hardcoded `None`)

## Git Remotes
- `origin` → upstream `open-world-project/model-router`
- `fork` → personal fork `mdgld/hermes-model-router` (push changes here)

## Per-Call Task-Aware Routing
- **Overview**: Optimizes LLM cost by routing mid-turn mechanical tool calls (e.g. read_file, search_files) to cheaper model tiers (the "floor tier"), while preserving the capability of high-reasoning models (the "working tier") for planning, writing/executing code, error recovery, and final synthesis.
- **Design Rationale**:
  - A floor tier protects the synthesis step. Each task defines a complexity band `[floor_tier, working_tier]`.
  - Detection uses a free, zero-LLM heuristic: read-only tool streaks drop the active tier to `floor_tier`, while write/execution/delegation tools or tool errors immediately restore the tier to `working_tier`.
- **Lag-by-One Invariant**: Switching `agent.model`/provider in `on_post_tool_call` takes effect on the *next* LLM call in the loop, which matches how the existing error-escalation system works.
- **Read-Only Tools**: `_READ_ONLY_TOOLS` contains read-only tools: `read_file`, `view_file`, `list_dir`, `grep_search`, `search_files`, `web_search`, `web_extract`, `x_search`, `session_search`, `read_terminal`, `skills_list`, `skill_view`, `vision_analyze`, `video_analyze`, `memory`, `read_resource`, `list_resources`, `read_url_content`, and `list_permissions`.

