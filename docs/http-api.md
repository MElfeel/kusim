# HTTP API

The command line of kusim over HTTP (FastAPI), for agents and services that have no shell. Code: `src/kusim/api.py`. It
adds no new behaviour: every command is a `POST` whose JSON body holds the options of the command, and the answer is the
JSON object the command prints (`docs/json-format.md`, `AGENTS.md`). Interactive documentation of every body is at `/docs`
(OpenAPI at `/openapi.json`).

```
pip install kusim[api]
kusim serve --workspace ./work --port 8000 --token "$KUSIM_API_TOKEN"
```

## How it works

- **A workspace folder** on the server holds the documents. Store one with `PUT /v1/files/model.json` (any valid JSON,
  at most 50 MB), name it in a command (`{"model": "model.json"}`), fetch results with `GET /v1/files/runs/a/results.json`.
  Names are relative, made of letters, digits, `.`, `_`, `-`, separated by `/`; `..`, absolute paths and links out of the
  workspace are refused (`422`, `"code": "bad_path"`). `GET /v1/files?prefix=runs/` lists, `DELETE /v1/files/runs` removes
  a file or a folder.
- **One process per command.** Each request runs `python -m kusim ...` in the workspace. A request cannot corrupt another,
  a runaway run is killed at the timeout (with its worker processes), and the answers are exactly those of the command line.
  The price is the start of Python with NumPy, SciPy and pandas: **about one second per command** on top of the run itself.
  For thousands of tiny runs use the Python package or one `ensemble`/`gradient` command.
- **Long commands** (`ensemble`, big runs): add `?async=true` to the `POST`. The answer is `202` with a job id;
  `GET /v1/jobs/{id}` gives `status` (`running`, `done`, `failed`, `timeout`), `http_status` and the `answer`. Jobs are kept
  in memory for an hour after they end and are lost when the server restarts. There is no cancel yet (the timeout stops
  them).

## Endpoints

| | |
|---|---|
| `GET /v1/health` | liveness (needs no token) |
| `GET /v1/limits` | the server's timeout, ceilings and concurrency |
| `GET /v1/schema/{name}` | JSON Schema: `model`, `schedule`, `observations`, `results`, `state`, `parameters`, `controls`, `prior` |
| `PUT / GET / DELETE /v1/files/{name}`, `GET /v1/files` | the workspace |
| `POST /v1/commands/{name}` | `template`, `validate`, `edit`, `diff`, `run`, `misfit`, `gradient`, `ensemble`, `import-deck`, `import-history`, `history-template` |
| `GET /v1/jobs`, `GET /v1/jobs/{id}` | background jobs |

`PUT /v1/files/{name}` checks that a document is JSON, except for the names ending in `.DATA`, `.INC`, `.csv`, `.txt`, `.xlsx`
and `.xlsm`, which are stored as they are: upload a deck (and the files it includes, next to it) or a workbook and name it in
`import-deck` or `import-history`.

Not in the API: `convert` (it reads a legacy `INPUT.DAT`, not JSON), `schema` (use the GET above), `version`, `serve`. A test
keeps the options of the API bodies and of the command line identical.

Body fields are the options of the command with underscores: `kusim run model.json --out runs/a --until 200 --max-cells 20000`
is

```
curl -X POST localhost:8000/v1/commands/run -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"model": "model.json", "out": "runs/a", "until": 200, "max_cells": 20000}'
```

A switch is a boolean (`"strict": true`), a repeated option is a list (`"set": ["kx=150", "dt=0.5"]` for `kusim edit`), file
options are workspace names. Unknown fields are an error (`"code": "unknown_key"`, with the field in `path`), like everywhere
in kusim.

## Status codes

| Status | Meaning |
|---|---|
| `200` | the command answered with `"ok": true`. A run that is not `healthy` is also `200`: read the body (`--strict` has no effect on the status) |
| `202` | a background job was started |
| `401` | no or wrong bearer token (`"code": "unauthorized"`) |
| `404` | a file or job does not exist |
| `413` | a stored document is over the size limit |
| `422` | bad input: the body has `code`, `path` and the hints of the command line (`did_you_mean`, `allowed`, ...) |
| `429` | the server is busy (`"code": "busy"`): `max_concurrent` commands are running; try again shortly |
| `500` | the command did not behave (`"code": "internal_error"`, with the tail of its error output) |
| `504` | the command ran longer than the server's timeout and was stopped (`"code": "timeout"`) |

## Safety

A network service is not a shell, so the server defends itself and its operator:

- **Ceilings for the guardrails.** The checks before a run (`docs/units-and-assumptions.md`) have limits that the CLI user can
  raise. Here the operator sets ceilings (`--max-cells`, `--max-seconds`, `--max-output-mb`, `--max-workers`); a client may
  lower a limit but not raise it or switch it off (`422`, `"code": "limit_exceeds_server_ceiling"`).
- **Timeout** per command (`--timeout`, default 900 s), **concurrency limit** (`--max-concurrent`, default 2).
- **Bearer token** (`--token` or `KUSIM_API_TOKEN`), compared in constant time. `health` is open.
- **No path leaks**: the commands run in the workspace with relative names, so errors do not show server paths.
- **Local by default**: it binds `127.0.0.1`. There is **no TLS, no user accounts, no rate limit per client and no quota** on
  the workspace. Do not expose it to the internet as it is: put it behind a reverse proxy that does TLS and authentication,
  and give the workspace its own disk quota. The workspace is shared by everyone who has the token.

## Limits

- One shared workspace, no users or projects; jobs live in memory.
- No streaming of progress and no cancel of a running job.
- A process per command (about one second of overhead each).
- Memory is not limited (the adjoint stores the whole run); use the operating system or the container for that.
