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
