"""agent_lab: a small, readable agent that drives kusim with an LLM of your choice (used by the second tutorial).

An agent is a loop: the model looks at the conversation and either asks for a *tool call* or gives a final answer; the
program runs the tool, adds the answer to the conversation and asks again. Everything here is plain Python and the standard
library, so you can read all of it:

1. the model connection  ``connect()``: one ``chat(messages, tools)`` function for Mistral, Groq, OpenRouter, Ollama (all
                         "OpenAI-style"), Anthropic, or a ``scripted`` model that replays a plan (no key, no internet);
2. the tools             ``kusim_tools()``: made from the ``kusim`` command line (every command is a tool, with its options
                         and help text), plus ``list_files``, ``read_file`` and ``write_file`` inside one workspace folder;
3. the loop              ``Agent(...).run(task)``: checks every call, asks you before the expensive or irreversible ones,
                         shortens long answers, writes ``trace.jsonl``, and stops after ``max_steps``.

Settings come from environment variables (the key is never written in a notebook):

    KUSIM_LLM_PROVIDER   mistral (default) | openai | anthropic | groq | openrouter | ollama | custom | scripted
    KUSIM_LLM_MODEL      a model name of the provider (each provider has a default below; check the provider's list)
    KUSIM_LLM_API_KEY    the key (or the provider's own variable: MISTRAL_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, ...)
    KUSIM_LLM_BASE_URL   another OpenAI-style server (LM Studio, vLLM, Gemini, ...), with KUSIM_LLM_PROVIDER=custom
    KUSIM_LLM_TIMEOUT    seconds to wait for one answer (default 120; 600 for ollama: a local model can be slow)
    KUSIM_LLM_TEMPERATURE  a number, or "none" to send no temperature (OpenAI's newer models accept only their default)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ====================================================================================================================
# 1. The model connection
# ====================================================================================================================


class LLMError(RuntimeError):
    """The model could not be reached or refused the request. The message says what to do."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    error: Optional[str] = None            # set when the model's arguments were not valid JSON


@dataclass
class Reply:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)       # input_tokens, output_tokens


# Defaults checked when this file was written; providers retire models and change free quotas often: look at the
# provider's console and set KUSIM_LLM_MODEL when a default stops working. ``min_interval`` is the least number of
# seconds between two requests (free tiers allow only a few requests per minute).
PRESETS = {
    "mistral": dict(kind="openai", base_url="https://api.mistral.ai/v1", key_env="MISTRAL_API_KEY", model="mistral-small-latest",
                    min_interval=1.5, signup="https://console.mistral.ai (free 'Experiment' plan)"),
    "groq": dict(kind="openai", base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY", model="llama-3.3-70b-versatile",
                 min_interval=2.0, signup="https://console.groq.com (free plan, limits per model)"),
    "openrouter": dict(kind="openai", base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY",
                       model="meta-llama/llama-3.3-70b-instruct:free", min_interval=3.0, signup="https://openrouter.ai"),
    "openai": dict(kind="openai", base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY", model="gpt-4.1-mini", min_interval=0.0,
                   temperature=None, signup="https://platform.openai.com (paid: add credit and set a spending limit)"),
    "ollama": dict(kind="openai", base_url="http://localhost:11434/v1", key_env=None, model="qwen3-16k", min_interval=0.0, timeout=600.0,
                   signup="https://ollama.com (runs on your own computer; pull a model that supports tools)"),
    "anthropic": dict(kind="anthropic", base_url="https://api.anthropic.com/v1", key_env="ANTHROPIC_API_KEY",
                      model="claude-sonnet-5-5", min_interval=0.0, signup="https://console.anthropic.com"),
    "custom": dict(kind="openai", base_url=None, key_env=None, model=None, min_interval=0.0, signup=""),
}


def parse_arguments(raw):
    """The arguments of a tool call as a dict, or ``(None, why)``. Models send a JSON string (or, rarely, a dict)."""
    if isinstance(raw, dict):
        return raw, None
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError) as error:
        return {}, f"the arguments are not valid JSON ({error}): {str(raw)[:120]!r}"
    if not isinstance(value, dict):
        return {}, f"the arguments must be a JSON object, not {type(value).__name__}"
    return value, None


class HttpLLM:
    """Base of the network models: one POST with retries. A free tier answers 429 when you ask too fast or the day's quota
    is used; we wait (``Retry-After`` if given) a few times, then say what to do."""

    def __init__(self, base_url, model, api_key=None, min_interval=0.0, timeout=120.0, max_retries=4):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.min_interval, self.timeout, self.max_retries = min_interval, timeout, max_retries
        self._last_call = 0.0

    def _headers(self):
        return {"Content-Type": "application/json"}

    def _post(self, path, body):
        wait = self.min_interval - (time.monotonic() - self._last_call)       # the pacer
        if wait > 0:
            time.sleep(wait)
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            self._last_call = time.monotonic()
            request = urllib.request.Request(self.base_url + path, data, self._headers(), method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                text = error.read().decode("utf8", "replace")[:300]
                if error.code in (401, 403):
                    raise LLMError(f"the provider refused the key ({error.code}): check the key and the model name. {text}")
                if error.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    delay = min(float(error.headers.get("Retry-After") or 2 ** (attempt + 1)), 60.0)
                    print(f"  (model busy, {error.code}: waiting {delay:g} s)")
                    time.sleep(delay)
                    continue
                if error.code == 429:
                    raise LLMError("rate limit or daily quota reached and waiting did not help: try again later, choose another "
                                   f"provider (KUSIM_LLM_PROVIDER), or use 'scripted'. {text}")
                raise LLMError(f"the model answered {error.code}: {text}")
            except urllib.error.URLError as error:
                raise LLMError(f"cannot reach {self.base_url}: {error.reason}. Check the internet connection (or that the "
                               "local server is running).")


class OpenAIStyleLLM(HttpLLM):
    """``POST {base_url}/chat/completions`` with tools: Mistral, Groq, OpenRouter, Ollama, LM Studio, OpenAI, ..."""

    def __init__(self, *args, temperature=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature                 # None: send none (some models accept only their default)

    def _headers(self):
        headers = super()._headers()
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def chat(self, messages, tools):
        body = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        data = self._post("/chat/completions", body)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected answer from the model: {str(data)[:300]}")
        calls = []
        for n, call in enumerate(message.get("tool_calls") or []):
            arguments, error = parse_arguments(call["function"].get("arguments"))
            calls.append(ToolCall(call.get("id") or f"call_{n}", call["function"]["name"], arguments, error))
        usage = data.get("usage") or {}
        return Reply(message.get("content") or "", calls,
                     {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)})


class AnthropicLLM(HttpLLM):
    """``POST {base_url}/messages`` (the Anthropic API is not OpenAI-style). The conversation is kept in the OpenAI style
    everywhere else in this file; this class translates it: ``tool_calls`` become ``tool_use`` blocks and the ``tool`` messages
    that follow become one ``tool_result`` user message."""

    def _headers(self):
        return {**super()._headers(), "x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}

    @staticmethod
    def convert(messages):
        system, out = [], []
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append(message["content"])
            elif role == "user":
                out.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                blocks = [{"type": "text", "text": message["content"]}] if message.get("content") else []
                for call in message.get("tool_calls") or []:
                    arguments, _ = parse_arguments(call["function"]["arguments"])
                    blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": arguments})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no text)"}]})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)                  # the results of one turn go in one message
                else:
                    out.append({"role": "user", "content": [block]})
        return "\n\n".join(system), out

    def chat(self, messages, tools):
        system, converted = self.convert(messages)
        body = {"model": self.model, "max_tokens": 2048, "system": system, "messages": converted}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        data = self._post("/messages", body)
        text, calls = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text.append(block["text"])
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        usage = data.get("usage") or {}
        return Reply("\n".join(text), calls, {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)})


class ScriptedLLM:
    """A model that replays a plan: ``script`` is a list of turns, each ``{"text": ..., "tool_calls": [{"name": ..., "arguments":
    {...}}, ...]}``. It never looks at the conversation, so it needs no key and no internet and gives the same run every time
    (the notebook uses it when no key is set, and the tests use it). A turn may also be a function ``turn(messages) -> dict``
    that reads the tool answers so far (to write the numbers in its final answer). ``from_trace`` replays what a real model did."""

    model = "scripted"

    def __init__(self, script):
        self.script, self.turn = list(script), 0

    @classmethod
    def from_trace(cls, path):
        events = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
        return cls([{"text": e["text"], "tool_calls": e["tool_calls"]} for e in events if e.get("event") == "assistant"])

    def chat(self, messages, tools):
        if self.turn >= len(self.script):
            return Reply("(the script has no more turns)")
        step = self.script[self.turn]
        step = step(messages) if callable(step) else step
        self.turn += 1
        calls = [ToolCall(f"call_{self.turn}_{n}", c["name"], dict(c.get("arguments", {}))) for n, c in enumerate(step.get("tool_calls", []))]
        return Reply(step.get("text", ""), calls)


def connect(provider=None, model=None, api_key=None, base_url=None, script=None):
    """The model to give to ``Agent``. With no arguments the environment variables decide (see the top of this file).
    ``script`` is for ``provider="scripted"``. A missing key is explained, not guessed."""
    provider = (provider or os.environ.get("KUSIM_LLM_PROVIDER") or "mistral").lower()
    if provider == "scripted":
        return ScriptedLLM(script or [])
    if provider not in PRESETS:
        raise LLMError(f"unknown provider {provider!r}: choose one of {', '.join(sorted(PRESETS))} or 'scripted'")
    preset = PRESETS[provider]
    model = model or os.environ.get("KUSIM_LLM_MODEL") or preset["model"]
    base_url = base_url or os.environ.get("KUSIM_LLM_BASE_URL") or preset["base_url"]
    key = api_key or os.environ.get("KUSIM_LLM_API_KEY") or (os.environ.get(preset["key_env"]) if preset["key_env"] else None)
    if not base_url or not model:
        raise LLMError("provider 'custom' needs KUSIM_LLM_BASE_URL and KUSIM_LLM_MODEL")
    if preset["key_env"] and not key:
        raise LLMError(f"no API key for {provider}: get one at {preset['signup']} and set {preset['key_env']} (or KUSIM_LLM_API_KEY) "
                       "before starting Jupyter; or use provider 'scripted' to run without a model")
    interval = float(os.environ.get("KUSIM_LLM_MIN_INTERVAL", preset["min_interval"]))
    timeout = float(os.environ.get("KUSIM_LLM_TIMEOUT", preset.get("timeout", 120.0)))
    if preset["kind"] == "anthropic":
        return AnthropicLLM(base_url, model, key, min_interval=interval, timeout=timeout)
    temperature = os.environ.get("KUSIM_LLM_TEMPERATURE")
    temperature = preset.get("temperature", 0.2) if temperature is None else (None if temperature.lower() == "none" else float(temperature))
    return OpenAIStyleLLM(base_url, model, key, min_interval=interval, timeout=timeout, temperature=temperature)


def ollama_models(base_url="http://localhost:11434"):
    """The names of the models installed in a running Ollama server, or ``None`` if it does not answer."""
    try:
        with urllib.request.urlopen(base_url + "/api/tags", timeout=3) as response:
            return [m["name"] for m in json.loads(response.read())["models"]]
    except (OSError, ValueError, KeyError):
        return None


def ollama_prepare(base="qwen3:8b", name="qwen3-16k", num_ctx=16384):
    """Make sure a copy of the installed model ``base`` called ``name`` exists with a context window of ``num_ctx`` tokens.
    Ollama's default window (4096 tokens) is smaller than the conversation of an agent: the model then loses the start of it.
    Runs ``ollama create`` (a few seconds, no download). Returns ``name``; raises ``LLMError`` with what to do if Ollama is not
    running or ``base`` is not installed."""
    installed = ollama_models()
    if installed is None:
        raise LLMError("Ollama does not answer on http://localhost:11434: install it from https://ollama.com and start it")
    if name in installed or name + ":latest" in installed:
        return name
    if base not in installed and base + ":latest" not in installed:
        raise LLMError(f"the model {base!r} is not installed: run  ollama pull {base}   (installed: {', '.join(installed) or 'none'})")
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        modelfile = Path(folder) / "Modelfile"
        modelfile.write_text(f"FROM {base}\nPARAMETER num_ctx {num_ctx}\n")
        done = subprocess.run(["ollama", "create", name, "-f", str(modelfile)], capture_output=True, text=True)
    if done.returncode:
        raise LLMError("ollama create failed: " + (done.stderr or done.stdout)[-300:])
    return name


# ====================================================================================================================
# 2. The tools: the kusim command line, as tool descriptions
# ====================================================================================================================

# Commands that are not tools: serving needs a person, convert/history-template are for people starting from files.
HIDDEN_COMMANDS = {"serve", "convert", "version", "history-template"}
# Options an agent does not get: the guardrail limits (an agent may not raise them), editing in place (write a new file), and
# --no-probe (it switches off the stability check that validate is for).
HIDDEN_OPTIONS = {"max_cells", "max_seconds", "max_output_mb", "in_place", "no_probe"}
# Options that are workspace file names; checked to stay inside the workspace.
FILE_OPTIONS = {"model", "input", "deck", "a", "b", "observations", "parameters", "prior", "schedule", "restart", "controls", "x",
                "patch", "output", "out", "state_out", "schedule_out", "wells_out", "controls_out", "wells_file"}
# Upper limits of numeric options (an agent may ask for less, not more).
MAXIMUM = {"workers": 4, "members": 100, "passes": 8, "check": 5}
# Tools that wait for a "yes" from the person: they take long, or write a conclusion.
NEEDS_APPROVAL = {"ensemble"}

LOCAL_TOOLS = [
    {"name": "list_files", "description": "List the files in a folder of the workspace (default: the workspace itself).",
     "parameters": {"type": "object", "properties": {"folder": {"type": "string", "description": "folder name, e.g. 'runs/a'"}}}},
    {"name": "read_file", "description": "Read a text file of the workspace (JSON files are shortened if long).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write a text file in the workspace (.json, .md or .txt). Overwriting an existing file, "
     "and any report.md, needs the person's approval.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
]


def _cli_commands():
    from kusim.cli import build_parser
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    helps = {a.dest: a.help for a in sub._choices_actions}
    return {name: (parser, helps.get(name) or "") for name, parser in sub.choices.items() if name not in HIDDEN_COMMANDS}


def _key(action):
    """The name of an option in a tool call: its longest flag without dashes (``--state-out`` is ``state_out``), which is also
    the name the HTTP API uses; a positional keeps its own name."""
    return max(action.option_strings, key=len).lstrip("-").replace("-", "_") if action.option_strings else action.dest


def _option_spec(action):
    """The JSON-schema entry of one argparse option."""
    if isinstance(action, argparse._StoreTrueAction):
        spec = {"type": "boolean"}
    elif isinstance(action, argparse._AppendAction):
        spec = {"type": "array", "items": {"type": "string"}}
    elif action.type is int:
        spec = {"type": "integer"}
    elif action.type is float:
        spec = {"type": "number"}
    else:
        spec = {"type": "string"}
    if action.choices:
        spec["enum"] = list(action.choices)
    if _key(action) in MAXIMUM:
        spec["maximum"] = MAXIMUM[_key(action)]
    if _key(action) in FILE_OPTIONS:
        spec["description"] = "a file name in the workspace. " + (action.help or "")
    else:
        spec["description"] = (action.help or "").replace("%%", "%")
    return spec


def kusim_tools():
    """``(tools, commands)``: ``tools`` is the list of tool descriptions for the model (name, description, JSON schema);
    ``commands`` maps a tool name to ``(command, [(dest, flag or None, is_switch, is_list)])`` to rebuild the command line.
    Made from ``kusim.cli.build_parser()``, so a new kusim option appears here without any change."""
    tools, commands = [], {}
    for command, (parser, summary) in _cli_commands().items():
        properties, required, layout = {}, [], []
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction) or _key(action) in HIDDEN_OPTIONS:
                continue
            properties[_key(action)] = _option_spec(action)
            if action.required or not action.option_strings:
                required.append(_key(action))
            flag = max(action.option_strings, key=len) if action.option_strings else None
            layout.append((_key(action), flag, isinstance(action, argparse._StoreTrueAction), isinstance(action, argparse._AppendAction)))
        name = command.replace("-", "_")
        tools.append({"name": name, "description": f"kusim {command}: {summary}",
                      "parameters": {"type": "object", "properties": properties, "required": required}})
        commands[name] = (command, layout)
    return tools + LOCAL_TOOLS, commands


def command_line(command, layout, arguments):
    """The argument list of ``kusim <command> ...`` for validated ``arguments``: positionals first, then options."""
    positional = [str(arguments[d]) for d, flag, _, _ in layout if flag is None and d in arguments]
    options = []
    for dest, flag, is_switch, is_list in layout:
        if flag is None or dest not in arguments or arguments[dest] in (None, False):
            continue
        if is_switch:
            options.append(flag)
        else:
            for value in (arguments[dest] if is_list else [arguments[dest]]):
                options += [flag, str(value)]
    return [command, *positional, *options]


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list}


def check_arguments(spec, arguments):
    """``None`` if the call is acceptable, else a message the model can act on (the command is then not run)."""
    properties = spec["parameters"]["properties"]
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return f"unknown argument(s) {unknown}; this tool takes {sorted(properties)}"
    missing = [k for k in spec["parameters"].get("required", []) if k not in arguments]
    if missing:
        return f"missing required argument(s) {missing}"
    for key, value in arguments.items():
        rule = properties[key]
        wanted = _TYPES[rule["type"]]
        if isinstance(value, bool) and rule["type"] != "boolean" or not isinstance(value, wanted):
            return f"argument {key!r} must be of type {rule['type']}, not {type(value).__name__}"
        if "enum" in rule and value not in rule["enum"]:
            return f"argument {key!r} must be one of {rule['enum']}"
        if "maximum" in rule and value > rule["maximum"]:
            return f"argument {key!r} may not exceed {rule['maximum']} here"
    return None


# ---------------------------------------------------------------------- the workspace and the transports
class Workspace:
    """One folder. Every file name the agent uses is relative to it and cannot leave it."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, name):
        if isinstance(name, str) and name.lstrip()[:1] in ("{", "["):
            raise ValueError("this must be the NAME of a file in the workspace, not JSON text (to change single values of a model use edit with set=['path=value']; to store JSON use write_file first)")
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ":" in name:
            raise ValueError(f"{name!r} is not a workspace file name: use a relative name like 'runs/a/model.json'")
        path = (self.root / name).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"{name!r} is outside the workspace")
        return path


class CliTransport:
    """Run ``python -m kusim ...`` in the workspace folder and read the one JSON object it prints."""

    def __init__(self, workspace, timeout=900.0):
        self.workspace, self.timeout = workspace, timeout

    def run(self, command, argv, arguments):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            done = subprocess.run([sys.executable, "-m", "kusim", *argv], cwd=self.workspace.root, capture_output=True, text=True,
                                  encoding="utf8", timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "code": "timeout", "error": f"the command took more than {self.timeout:g} s and was stopped"}
        try:
            return json.loads(done.stdout)
        except ValueError:
            return {"ok": False, "code": "no_json", "error": "the command printed no JSON: " + (done.stderr or done.stdout)[-300:]}


class HttpTransport:
    """The same commands over HTTP: ``POST /v1/commands/{name}`` on a ``kusim serve`` whose workspace is the same folder (so
    the file names mean the same thing). ``HttpTransport.start(workspace)`` runs the server for you (needs ``pip install
    kusim[api]``); stop it with ``close()``."""

    def __init__(self, base_url, token=None, process=None):
        self.base_url, self.token, self.process = base_url.rstrip("/"), token, process

    @classmethod
    def start(cls, workspace, port=8765, token="agent-lab-token"):
        process = subprocess.Popen([sys.executable, "-m", "kusim", "serve", "--workspace", str(workspace.root), "--port", str(port),
                                    "--token", token], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        transport = cls(f"http://127.0.0.1:{port}", token, process)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("the server did not start (is it installed? pip install kusim[api]): " + process.stderr.read()[-300:])
            try:
                urllib.request.urlopen(transport.base_url + "/v1/health", timeout=1).read()
                return transport
            except OSError:
                time.sleep(0.1)
        process.terminate()
        raise RuntimeError("the server did not answer in 10 s")

    def close(self):
        if self.process:
            self.process.terminate()

    def run(self, command, argv, arguments):
        body = {k: v for k, v in arguments.items() if v not in (None, False)}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        request = urllib.request.Request(f"{self.base_url}/v1/commands/{command}", json.dumps(body).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:                    # 422 is exit code 2: the body is the same error object
            try:
                return json.loads(error.read())
            except ValueError:
                return {"ok": False, "code": "http_error", "error": f"HTTP {error.code}"}


# ====================================================================================================================
# 3. Shortening answers: a model has a small window and a free tier a small token budget
# ====================================================================================================================

FIRST_KEYS = ("ok", "error", "code", "path", "hint", "did_you_mean", "healthy", "warnings")


def _shrink(value, items, chars):
    if isinstance(value, dict):
        keys = [k for k in FIRST_KEYS if k in value] + [k for k in value if k not in FIRST_KEYS]
        return {k: _shrink(value[k], items, chars) for k in keys}
    if isinstance(value, list):
        kept = [_shrink(v, items, chars) for v in value[:items]]
        return kept + [f"... {len(value) - items} more"] if len(value) > items else kept
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, str) and len(value) > chars:
        return value[:chars] + "..."
    return value


def compact(value, max_chars=3000):
    """JSON text of ``value`` no longer than about ``max_chars``: long lists are cut ("... 40 more"), long strings and
    floats are shortened, the keys that matter (ok, error, code, path, hint) come first."""
    for items, chars in ((12, 400), (6, 250), (3, 150)):
        text = json.dumps(_shrink(value, items, chars), separators=(",", ":"), ensure_ascii=False)
        if len(text) <= max_chars:
            return text
    return text[:max_chars] + f'... [cut: {len(text) - max_chars} characters more]'


# ====================================================================================================================
# 4. The loop
# ====================================================================================================================

SYSTEM_PROMPT = """You operate kusim, a 3-D two-phase (oil/water) reservoir simulator, for a student. Use the tools; do not guess numbers.

Rules:
- Work only with the files of the workspace; never invent file names, list them if unsure. Write new files, do not overwrite.
- Units are oilfield units: ft, md, psi, cp, STB/day, days. Rates are positive for producers and injectors. Block numbers count from 1.
- Validate a model before you run it. If validate warns about dt (dt_too_large), use the suggested dt.
- An error answer has a path, a code and hints (did_you_mean, allowed): fix the input with them, do not repeat the same call.
- A run can finish with ok=true and still be wrong: check `healthy` and the warnings before you use a number, and report them.
- A good match to the history does not prove the permeability field is right: say "reproduces the history" and check on data
  that were not used for the calibration.
- When you import data, read `dropped` and `defaults_used` in the answer and tell the student what they mean.
- Do not match the well rates that you impose (match pressures and water cuts).
- Be short. Finish with a plain answer: what you did, the numbers, what you did not check."""


def system_prompt(full=False):
    """``SYSTEM_PROMPT``; with ``full=True`` the whole ``AGENTS.md`` of the repository is added (about 2500 tokens: a small
    free-tier model may do better without it)."""
    agents = Path(__file__).resolve().parent.parent / "AGENTS.md"
    if full and agents.exists():
        return SYSTEM_PROMPT + "\n\n--- AGENTS.md ---\n" + agents.read_text(encoding="utf8")
    return SYSTEM_PROMPT


def ask_user(name, arguments, why):
    """Default approval: show the call and ask on the keyboard (a notebook shows a text box)."""
    print(f"\n  APPROVAL NEEDED: {name}({json.dumps(arguments)[:300]}) - {why}")
    return input("  Allow? [y/N] ").strip().lower() in ("y", "yes")


def auto_approve(name, arguments, why):
    print(f"  (approval for {name} given automatically)")
    return True


def deny_all(name, arguments, why):
    return False


@dataclass
class Run:
    task: str
    answer: str
    steps: int
    stopped: str                      # "answered", "max_steps"
    tokens: Dict[str, int]
    trace: Path
    calls: List[dict]


class Agent:
    """``Agent(llm, workspace).run("task")``. ``extra_tools={"name": (description, function)}`` adds your own tools. ``llm`` is from ``connect()``; ``workspace`` a folder (created). ``transport`` is
    ``"cli"`` or an ``HttpTransport``. ``approve(name, arguments, why) -> bool`` is asked before ``NEEDS_APPROVAL`` tools,
    before overwriting a file and before writing ``report.md``. Each event is appended to ``trace.jsonl`` in the workspace."""

    def __init__(self, llm, workspace, transport="cli", approve=ask_user, max_steps=12, system=None, verbose=True, trace="trace.jsonl",
                 max_chars=3000, extra_tools=None, one_call_at_a_time=True):
        self.llm, self.workspace = llm, Workspace(workspace) if not isinstance(workspace, Workspace) else workspace
        self.transport = CliTransport(self.workspace) if transport == "cli" else transport
        self.approve, self.max_steps, self.verbose, self.max_chars = approve, max_steps, verbose, max_chars
        self.one_call_at_a_time = one_call_at_a_time                # small models send dependent calls together: run only the first
        self.system = system or system_prompt()
        self.tools, self.commands = kusim_tools()
        self.extra = dict(extra_tools or {})                       # name -> (tool description, function(**arguments) -> dict)
        self.tools = self.tools + [spec for spec, _ in self.extra.values()]
        self.specs = {t["name"]: t for t in self.tools}
        self.trace_path = self.workspace.root / trace

    # -- the trace
    def _log(self, **event):
        with open(self.trace_path, "a", encoding="utf8") as handle:
            handle.write(json.dumps({"time": round(time.time(), 1), **event}, ensure_ascii=False) + "\n")

    def _say(self, text):
        if self.verbose:
            try:
                print(text)
            except UnicodeEncodeError:                                  # a console that cannot show what the model wrote
                print(text.encode("ascii", "replace").decode())

    # -- one tool call
    def _needs_approval(self, name, arguments):
        if name in NEEDS_APPROVAL:
            return "this run takes a while (an ensemble of many simulations)"
        if name == "write_file":
            try:
                path = self.workspace.resolve(arguments["path"])
            except (ValueError, KeyError):
                return None
            if path.exists():
                return "it would overwrite an existing file"
            if path.name.lower().startswith("report"):
                return "it writes a conclusion under your name"
        return None

    def execute(self, call, seen):
        """The answer (a dict) to one tool call. The command runs only if the call is valid and, where needed, approved."""
        spec = self.specs.get(call.name)
        if spec is None:
            return {"ok": False, "code": "unknown_tool", "error": f"there is no tool {call.name!r}", "tools": sorted(self.specs)}
        if call.error:
            return {"ok": False, "code": "bad_tool_arguments", "error": call.error + " (nothing was run)"}
        problem = check_arguments(spec, call.arguments)
        if problem:
            return {"ok": False, "code": "invalid_arguments", "error": problem + " (nothing was run)"}
        key = json.dumps([call.name, call.arguments], sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            return {"ok": False, "code": "repeated_call", "error": "you made this exact call three times: change the input or give your answer"}
        try:                                                            # every file name must stay inside the workspace
            for dest, value in call.arguments.items():
                if call.name not in self.extra and (dest in FILE_OPTIONS or dest in ("path", "folder")):
                    self.workspace.resolve(value)
        except ValueError as error:
            return {"ok": False, "code": "bad_path", "error": str(error)}
        why = self._needs_approval(call.name, call.arguments)
        if why and not self.approve(call.name, call.arguments, why):
            self._log(event="declined", tool=call.name, arguments=call.arguments)
            return {"ok": False, "code": "declined", "error": "the student declined this call: do not repeat it; ask what they prefer"}
        if call.name in self.commands:
            command, layout = self.commands[call.name]
            return self.transport.run(command, command_line(command, layout, call.arguments), call.arguments)
        if call.name in self.extra:
            return self.extra[call.name][1](**call.arguments)
        return getattr(self, "_tool_" + call.name)(**call.arguments)

    def call(self, name, **arguments):
        """Make one tool call yourself (what the model does in the loop), with the same checks. Returns the answer dict."""
        return self.execute(ToolCall("by_hand", name, arguments), {})

    # -- the three local tools
    def _tool_list_files(self, folder="."):
        try:
            root = self.workspace.resolve(folder)
        except ValueError as error:
            return {"ok": False, "code": "bad_path", "error": str(error)}
        if not root.is_dir():
            return {"ok": False, "code": "not_a_folder", "error": f"{folder!r} is not a folder of the workspace"}
        files = sorted(p.relative_to(self.workspace.root).as_posix() + ("/" if p.is_dir() else "") for p in root.iterdir()
                       if not p.name.startswith(("trace", ".")))
        return {"ok": True, "folder": folder, "files": files}

    def _tool_read_file(self, path):
        target = self.workspace.resolve(path)
        if not target.is_file():
            return {"ok": False, "code": "no_such_file", "error": f"{path!r} does not exist (use list_files)"}
        try:
            text = target.read_text(encoding="utf8")
        except UnicodeDecodeError:
            return {"ok": False, "code": "not_text", "error": f"{path!r} is not a text file"}
        if target.suffix == ".json":
            try:
                return {"ok": True, "path": path, "content": json.loads(text)}
            except ValueError:
                pass
        return {"ok": True, "path": path, "content": text[:self.max_chars], "truncated": len(text) > self.max_chars}

    def _tool_write_file(self, path, content):
        target = self.workspace.resolve(path)
        if target.suffix.lower() not in (".json", ".md", ".txt"):
            return {"ok": False, "code": "bad_extension", "error": "write .json, .md or .txt files only"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf8")
        return {"ok": True, "written": path, "characters": len(content)}

    # -- the loop
    def run(self, task):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        tokens, calls, seen, answer, stopped = {"input_tokens": 0, "output_tokens": 0}, [], {}, "", "max_steps"
        self._log(event="task", task=task, model=getattr(self.llm, "model", "?"))
        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(messages, self.tools)
            for k in tokens:
                tokens[k] += reply.usage.get(k, 0)
            self._log(event="assistant", step=step, text=reply.text,
                      tool_calls=[{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls], usage=reply.usage)
            if reply.text:
                self._say(f"\n[{step}] agent: {reply.text.strip()}")
            if not reply.tool_calls:
                answer, stopped = reply.text, "answered"
                break
            messages.append({"role": "assistant", "content": reply.text or None,
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in reply.tool_calls]})
            for position, call in enumerate(reply.tool_calls):
                shown = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
                self._say(f"[{step}] -> {call.name}({shown if len(shown) <= 160 else shown[:157] + '...'})")
                if position and self.one_call_at_a_time:                  # the later calls may depend on the first one
                    result = {"ok": False, "code": "not_run", "error": "one tool call at a time: this call was not run; make it again after you have seen the answer of the previous one"}
                else:
                    result = self.execute(call, seen)
                text = compact(result, self.max_chars)
                calls.append({"step": step, "tool": call.name, "arguments": call.arguments, "ok": result.get("ok"), "code": result.get("code")})
                self._log(event="tool", step=step, tool=call.name, arguments=call.arguments, ok=result.get("ok"), result=text)
                self._say(f"      <- ok={result.get('ok')}" + (f" {result.get('code')}: {str(result.get('error'))[:140]}" if not result.get("ok") else ""))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        else:
            answer = "(stopped: the step limit was reached before the agent gave an answer)"
        self._log(event="end", stopped=stopped, tokens=tokens)
        self._say(f"\n[{stopped}] {len(calls)} tool calls, {tokens['input_tokens']} tokens in / {tokens['output_tokens']} out")
        return Run(task, answer, len(calls), stopped, tokens, self.trace_path, calls)


def read_trace(path):
    """The events of a ``trace.jsonl`` as a list of dicts."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


if __name__ == "__main__":                       # python agent_lab.py : check the connection with one tool call
    parser = argparse.ArgumentParser(description="Test the model connection and list the tools.")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    args = parser.parse_args()
    tools, _ = kusim_tools()
    print("tools:", ", ".join(t["name"] for t in tools))
    llm = connect(args.provider, args.model)
    answer = llm.chat([{"role": "user", "content": "Call the tool list_files once, with no arguments."}], [t for t in tools if t["name"] == "list_files"])
    print("model:", getattr(llm, "model", "?"), "| asked for:", [(c.name, c.arguments) for c in answer.tool_calls] or answer.text)
