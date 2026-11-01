# An LLM agent that drives kusim (tutorial 2)

`examples/tutorial_agentic_workflow.ipynb` shows a program that takes a request in plain English, calls `kusim` commands, reads
the JSON answers, fixes its mistakes and reports. The code is `examples/agent_lab.py` (standard library only, readable in an
afternoon), the task files `agent_lab_case.py` and the tasks and scripted plans `agent_lab_scripts.py`. `AGENTS.md` is the
instruction sheet any agent should get; `agent_lab.SYSTEM_PROMPT` is a short version of it.

## Choose a model

Set environment variables **before starting Jupyter** (PowerShell: `$env:MISTRAL_API_KEY = "..."`):

| Provider (`KUSIM_LLM_PROVIDER`) | Key | Default model | Note |
|---|---|---|---|
| `mistral` (default) | `MISTRAL_API_KEY` | `mistral-small-latest` | free "Experiment" plan at console.mistral.ai; low requests-per-minute; prompts may be used for training |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | free plan, limits per model |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | its own protocol (a small adapter, `AnthropicLLM`) |
| `openrouter` | `OPENROUTER_API_KEY` | a free Llama | |
| `ollama` | none | `llama3.1` | on your computer; the model must support tools |
| `custom` | `KUSIM_LLM_API_KEY` | `KUSIM_LLM_MODEL` | any OpenAI-style server: also `KUSIM_LLM_BASE_URL` |
| `scripted` | none | none | replays a plan: no key, no internet, same result every time |

`KUSIM_LLM_MODEL` overrides the model; `KUSIM_LLM_MIN_INTERVAL` the seconds between requests. **Free quotas and model names change
often; the defaults were not tested against the live services when written** (the tests use a local fake server). Check with
`py examples/agent_lab.py --provider mistral`, which makes one tool-call request.

Without a key the notebook uses the scripted plans and runs completely. Troubleshooting: "refused the key" (401): wrong key or
model name; "rate limit or daily quota" (429 after retries): wait, change provider, or use `scripted`; a small model that
sends invalid arguments gets the error back (`invalid_arguments`) and the call is not run; one that repeats a call three times is stopped.

## What the agent may and may not do

- Tools are **made from the command line** (`kusim.cli.build_parser()`): every command except `serve`, `convert`, `version` and
  `history-template`, with its options and help text, plus `list_files`, `read_file`, `write_file` in one workspace folder.
  Add a kusim option and the agent has it. Your own tool: `Agent(..., extra_tools={"name": (description, function)})`.
- Not offered: the guardrail limits, `--in-place`, files outside the workspace. Caps: 4 workers, 100 members, 8 passes.
- The person is asked before `ensemble`, before overwriting a file and before writing `report.md` (`approve=ask_user`; `auto_approve`
  and `deny_all` for scripts and tests). Every call and answer is appended to `trace.jsonl` (`ScriptedLLM.from_trace` replays it).
- Transport: `Agent(..., transport="cli")` runs `python -m kusim` in the workspace; `HttpTransport.start(workspace)` uses
  `kusim serve` on the same folder (needs `kusim[api]`). Same tools, same answers.
- Answers are shortened for the model (`compact`): long lists cut, floats rounded, `ok`/`error`/`code`/`hint` first.

## Limits

No live model has been run through the tasks yet; the committed notebook output is the scripted run. A real model will take other
routes, sometimes worse ones. Run the notebook with your key and read the trace. Runs are not deterministic.
