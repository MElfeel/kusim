# An LLM agent that drives kusim (tutorial 2)

Three versions of the notebook, the same tasks: `tutorial_agentic_workflow_local.ipynb` (Ollama, `qwen3:8b`, no key, nothing leaves
your computer, minutes per task), `tutorial_agentic_workflow_frontier.ipynb` (Anthropic or OpenAI, a paid key, fast and reliable) and
`tutorial_agentic_workflow.ipynb` (provider-neutral: Mistral by default, any provider by environment variable). Each falls back to
scripted plans when no model is available. The notebook shows a program that takes a request in plain English, calls `kusim` commands, reads
the JSON answers, fixes its mistakes and reports. The code is `examples/agent_lab.py` (standard library only, readable in an
afternoon), the task files `agent_lab_case.py` and the tasks and scripted plans `agent_lab_scripts.py`. `AGENTS.md` is the
instruction sheet any agent should get; `agent_lab.SYSTEM_PROMPT` is a short version of it.

## Choose a model

Set environment variables **before starting Jupyter** (PowerShell: `$env:MISTRAL_API_KEY = "..."`):

| Provider (`KUSIM_LLM_PROVIDER`) | Key | Default model | Note |
|---|---|---|---|
| `mistral` (default) | `MISTRAL_API_KEY` | `mistral-small-latest` | free "Experiment" plan at console.mistral.ai; low requests-per-minute; prompts may be used for training |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | free plan, limits per model |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | paid; its own protocol (a small adapter, `AnthropicLLM`); `claude-haiku-5-5` is cheaper, `claude-opus-5-5` stronger |
| `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini` | paid; the default name is not tested against the live service, check OpenAI's models page; no temperature is sent (newer models accept only the default) |
| `openrouter` | `OPENROUTER_API_KEY` | a free Llama | |
| `ollama` | none | `llama3.1` | on your computer; the model must support tools |
| `custom` | `KUSIM_LLM_API_KEY` | `KUSIM_LLM_MODEL` | any OpenAI-style server: also `KUSIM_LLM_BASE_URL` |
| `scripted` | none | none | replays a plan: no key, no internet, same result every time |

`KUSIM_LLM_MODEL` overrides the model; `KUSIM_LLM_MIN_INTERVAL` the seconds between requests; `KUSIM_LLM_TIMEOUT` the seconds to wait for an answer (600 for `ollama`); `KUSIM_LLM_TEMPERATURE` a number or `none`. **Free quotas and model names change
often; the defaults were not tested against the live services when written** (the tests use a local fake server). Check with
`py examples/agent_lab.py --provider mistral`, which makes one tool-call request.

Without a key the notebook uses the scripted plans and runs completely. Troubleshooting: "refused the key" (401): wrong key or
model name; "rate limit or daily quota" (429 after retries): wait, change provider, or use `scripted`; a small model that
sends invalid arguments gets the error back (`invalid_arguments`) and the call is not run; one that repeats a call three times is stopped.

## A local model with Ollama (no key)

Measured on one Windows laptop (Task A and B of the notebook, `ollama` 0.18, approvals denied, one run each):

| Model | Result |
|---|---|
| `llama3.2` (3B) | calls tools but with wrong arguments; all calls failed |
| `mistral:latest` (7B) | never calls a tool (writes shell commands as text) |
| `qwen3:8b` | Task A and Task B completed (right numbers on B), 2.5 to 7 minutes each; it also invented a `schedule.json` and, on A, compared at day 365 instead of the end; read its answers critically |

**Ollama's default context window is small (4096 tokens), and the agent's conversation does not fit**: the model then loses the start
of the conversation and may answer nothing. Make a copy with a larger window and use it:

```
printf 'FROM qwen3:8b\nPARAMETER num_ctx 16384\n' > Modelfile   (or create the file by hand)
ollama create qwen3-16k -f Modelfile
KUSIM_LLM_PROVIDER=ollama   KUSIM_LLM_MODEL=qwen3-16k      (the local notebook does this step for you: `agent_lab.ollama_prepare`)
```

Local models need minutes per task and are weaker than hosted ones. `Agent` runs one tool call per turn by default (small models
send dependent calls together); `one_call_at_a_time=False` allows several.

## What the agent may and may not do

- Tools are **made from the command line** (`kusim.cli.build_parser()`): every command except `serve`, `convert`, `version` and
  `history-template`, with its options and help text, plus `list_files`, `read_file`, `write_file` in one workspace folder.
  Add a kusim option and the agent has it. Your own tool: `Agent(..., extra_tools={"name": (description, function)})`.
- Not offered: the guardrail limits, `--in-place`, `--no-probe` (it switches off the stability check), files outside the workspace. Caps: 4 workers, 100 members, 8 passes.
- The person is asked before `ensemble`, before overwriting a file and before writing `report.md` (`approve=ask_user`; `auto_approve`
  and `deny_all` for scripts and tests). Every call and answer is appended to `trace.jsonl` (`ScriptedLLM.from_trace` replays it).
- Transport: `Agent(..., transport="cli")` runs `python -m kusim` in the workspace; `HttpTransport.start(workspace)` uses
  `kusim serve` on the same folder (needs `kusim[api]`). Same tools, same answers.
- Answers are shortened for the model (`compact`): long lists cut, floats rounded, `ok`/`error`/`code`/`hint` first.

## Limits

No live model has been run through the tasks yet; the committed notebook output is the scripted run. A real model will take other
routes, sometimes worse ones. Run the notebook with your key and read the trace. Runs are not deterministic.

## "No API key" although you set one

An environment variable reaches only programs started **after** it was set, **from the same terminal** (`$env:OPENAI_API_KEY = "..."` in
PowerShell lasts for that window only). A Jupyter or VS Code that was already running, or started from the Start menu, does not see it, and
neither does a running kernel. Either set it and start `jupyter lab` from that terminal, or restart the kernel, or put the line
`OPENAI_API_KEY=sk-...` in a file called `.env` next to `agent_lab.py` (`agent_lab.load_env_file` reads it; variables already set win; the
file is not tracked by git), or type the key in the notebook's `getpass` cell. The variable names are `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`MISTRAL_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `KUSIM_LLM_API_KEY` for any provider.
