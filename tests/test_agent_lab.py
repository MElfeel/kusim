"""The agent of the second tutorial (examples/agent_lab.py): tools made from the command line, checks before a call, the loop,
the model connections (against a small local server, no network) and the three scripted tasks."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))
import agent_lab as al                      # noqa: E402
import agent_lab_scripts as scripts         # noqa: E402

from kusim.cli import main                  # noqa: E402


@pytest.fixture(scope="module")
def tools():
    return al.kusim_tools()


def test_tools_come_from_the_command_line(tools):
    specs, commands = tools
    names = {t["name"] for t in specs}
    assert {"validate", "edit", "run", "misfit", "gradient", "ensemble", "import_deck", "import_history", "read_file"} <= names
    assert not names & {"serve", "convert", "version"}
    for spec in specs:
        assert spec["description"] and spec["parameters"]["type"] == "object"
    edit = next(t for t in specs if t["name"] == "edit")["parameters"]
    assert edit["properties"]["set"]["type"] == "array" and "in_place" not in edit["properties"]
    run = next(t for t in specs if t["name"] == "run")["parameters"]["properties"]
    assert not {"max_cells", "max_seconds", "max_output_mb"} & set(run)           # the agent may not raise the guardrails
    validate = next(t for t in specs if t["name"] == "validate")["parameters"]["properties"]
    assert "no_probe" not in validate                                             # it would switch off the stability check
    assert "PATH=VALUE" in edit["properties"]["set"]["description"] and "FILE" in edit["properties"]["patch"]["description"]
    ensemble = next(t for t in specs if t["name"] == "ensemble")["parameters"]
    assert ensemble["properties"]["workers"]["maximum"] == 4 and "method" in ensemble["required"]
    assert al.command_line(*commands["edit"], {"model": "m.json", "set": ["dt=5", "kx=2"], "output": "o.json"}) == [
        "edit", "m.json", "--set", "dt=5", "--set", "kx=2", "--output", "o.json"]
    assert al.command_line(*commands["run"], {"model": "m.json", "strict": True, "csv": False}) == ["run", "m.json", "--strict"]


def test_arguments_are_checked_before_anything_runs(tools):
    spec = next(t for t in tools[0] if t["name"] == "ensemble")
    assert "unknown argument" in al.check_arguments(spec, {"model": "m", "mehtod": "esmda"})
    assert "missing required" in al.check_arguments(spec, {"model": "m"})
    assert "one of" in al.check_arguments(spec, {"model": "m", "method": "magic", "observations": "o", "parameters": "p", "prior": "q"})
    assert "may not exceed" in al.check_arguments(spec, {"model": "m", "method": "esmda", "observations": "o", "parameters": "p",
                                                         "prior": "q", "members": 5000})
    assert "type" in al.check_arguments(spec, {"model": "m", "method": "esmda", "observations": "o", "parameters": "p",
                                               "prior": "q", "members": True})
    assert al.check_arguments(spec, {"model": "m", "method": "esmda", "observations": "o", "parameters": "p", "prior": "q", "members": 10}) is None


def test_compact_shortens_but_keeps_what_matters():
    answer = {"data": list(range(500)), "ok": False, "error": "x" * 2000, "code": "bad", "value": 1 / 3}
    text = al.compact(answer, 600)
    assert len(text) <= 700 and text.startswith('{"ok":false,"error":')
    assert "more" in text and "0.333333" in al.compact({"v": 1 / 3})


def test_workspace_cannot_be_left(tmp_path):
    ws = al.Workspace(tmp_path)
    assert ws.resolve("runs/a/model.json") == (tmp_path / "runs/a/model.json").resolve()
    for bad in ("../x.json", "/etc/passwd", "C:/x.json", "a/../../x", "", None):
        with pytest.raises(ValueError):
            ws.resolve(bad)


def test_anthropic_conversion():
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "validate", "arguments": '{"model": "m.json"}'}},
                    {"id": "t2", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "{}"}, {"role": "tool", "tool_call_id": "t2", "content": "{}"}]
    system, converted = al.AnthropicLLM.convert(messages)
    assert system == "be brief" and [m["role"] for m in converted] == ["user", "assistant", "user"]
    assert converted[1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "validate", "input": {"model": "m.json"}}
    assert [b["tool_use_id"] for b in converted[2]["content"]] == ["t1", "t2"]           # both results in one user message


class FakeServer:
    """A local HTTP server that answers like a model API; ``answers`` are returned in turn, each as (status, headers, body)."""

    def __init__(self, answers):
        outer = self
        self.answers, self.requests = list(answers), []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, json.loads(body)))
                status, headers, payload = outer.answers.pop(0)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    servers = []

    def make(answers):
        servers.append(FakeServer(answers))
        return servers[-1]
    yield make
    for s in servers:
        s.close()


LIST_TOOL = [{"name": "list_files", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def test_openai_style_model_with_a_retry(fake):
    reply = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "abc", "type": "function", "function": {"name": "list_files", "arguments": '{"folder": "field"}'}},
        {"id": "def", "type": "function", "function": {"name": "read_file", "arguments": "{not json"}}]}}],
             "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
    server = fake([(429, {"Retry-After": "0"}, {"message": "slow down"}), (200, {}, reply)])
    llm = al.OpenAIStyleLLM(server.url, "some-model", "KEY")
    answer = llm.chat([{"role": "user", "content": "hi"}], LIST_TOOL)
    assert len(server.requests) == 2                                                       # the 429 was retried
    path, headers, body = server.requests[-1]
    assert path == "/chat/completions" and headers["authorization"] == "Bearer KEY"
    assert body["model"] == "some-model" and body["tools"][0]["function"]["name"] == "list_files"
    assert answer.tool_calls[0].arguments == {"folder": "field"} and answer.usage == {"input_tokens": 11, "output_tokens": 5}
    assert answer.tool_calls[1].error and "not valid JSON" in answer.tool_calls[1].error      # bad arguments are reported, not run


def test_anthropic_model(fake):
    reply = {"content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "tu1", "name": "list_files", "input": {}}],
             "usage": {"input_tokens": 7, "output_tokens": 3}}
    server = fake([(200, {}, reply)])
    llm = al.AnthropicLLM(server.url, "a-model", "SECRET")
    answer = llm.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], LIST_TOOL)
    path, headers, body = server.requests[0]
    assert path == "/messages" and headers["x-api-key"] == "SECRET" and "anthropic-version" in headers
    assert body["system"] == "s" and body["tools"][0]["input_schema"]["type"] == "object"
    assert answer.text == "ok" and answer.tool_calls[0].name == "list_files" and answer.usage["input_tokens"] == 7


def test_no_temperature_is_sent_when_none(fake):
    server = fake([(200, {}, {"choices": [{"message": {"content": "hi"}}]})])
    al.OpenAIStyleLLM(server.url, "m", "k", temperature=None).chat([{"role": "user", "content": "x"}], [])
    assert "temperature" not in server.requests[0][2]


def test_ollama_helpers(monkeypatch):
    assert al.ollama_models("http://127.0.0.1:1") is None                      # nothing listens there
    monkeypatch.setattr(al, "ollama_models", lambda *a, **k: None)
    with pytest.raises(al.LLMError, match="does not answer"):
        al.ollama_prepare()
    monkeypatch.setattr(al, "ollama_models", lambda *a, **k: ["other:latest"])
    with pytest.raises(al.LLMError, match="ollama pull qwen3:8b"):             # the base model is missing: say what to run
        al.ollama_prepare()
    monkeypatch.setattr(al, "ollama_models", lambda *a, **k: ["qwen3-16k:latest", "qwen3:8b"])
    assert al.ollama_prepare() == "qwen3-16k"                                  # already there: nothing is created


def test_key_from_a_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KUSIM_LLM_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(al, "env_folders", lambda: [tmp_path])                  # not the real .env files of whoever runs the tests
    with pytest.raises(al.LLMError, match="AFTER it was set"):                # the message explains the usual mistake
        al.connect("openai")
    (tmp_path / ".env").write_text('# my keys\nOPENAI_API_KEY="sk-test"\n')
    assert al.connect("openai").api_key == "sk-test"
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")                   # a variable that is already set wins
    assert al.connect("openai").api_key == "from-environment"


def test_a_refused_key_is_explained(fake):
    server = fake([(401, {}, {"error": "bad key"})])
    with pytest.raises(al.LLMError, match="refused the key"):
        al.OpenAIStyleLLM(server.url, "m", "wrong").chat([{"role": "user", "content": "hi"}], [])


def test_connect(monkeypatch):
    for name in ("KUSIM_LLM_PROVIDER", "KUSIM_LLM_MODEL", "KUSIM_LLM_API_KEY", "KUSIM_LLM_BASE_URL", "MISTRAL_API_KEY", "GROQ_API_KEY",
                 "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(al.LLMError, match="MISTRAL_API_KEY"):                  # Mistral is the default; the message says what to do
        al.connect()
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = al.connect()
    assert isinstance(llm, al.OpenAIStyleLLM) and llm.base_url == "https://api.mistral.ai/v1" and llm.model == "mistral-small-latest"
    monkeypatch.setenv("KUSIM_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(al.connect(), al.AnthropicLLM)
    assert isinstance(al.connect("scripted"), al.ScriptedLLM)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    openai = al.connect("openai")
    assert openai.base_url == "https://api.openai.com/v1" and openai.temperature is None      # newer OpenAI models accept only the default
    assert al.connect("ollama").timeout == 600.0 and al.connect("ollama").model == "qwen3-16k"
    monkeypatch.setenv("KUSIM_LLM_TEMPERATURE", "0.5")
    assert al.connect("openai").temperature == 0.5
    monkeypatch.setenv("KUSIM_LLM_TEMPERATURE", "none")
    assert al.connect("mistral", api_key="k").temperature is None
    with pytest.raises(al.LLMError, match="unknown provider"):
        al.connect("nope")
    with pytest.raises(al.LLMError, match="custom"):
        al.connect("custom")
    assert al.connect("custom", model="m", base_url="http://localhost:1/v1").base_url == "http://localhost:1/v1"


# ---------------------------------------------------------------------- the loop
@pytest.fixture
def work(tmp_path):
    main(["template", "-o", str(tmp_path / "model.json"), "--days", "20"])
    (tmp_path / "model.json").exists()
    return tmp_path


def run_script(work, script, **kwargs):
    kwargs.setdefault("approve", al.deny_all)
    kwargs.setdefault("verbose", False)
    return al.Agent(al.connect("scripted", script=script), work, **kwargs).run("task")


def call(name, **arguments):
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def test_loop_runs_validates_and_stops(work):
    run = run_script(work, [call("validate", model="model.json"), call("run", model="model.json", out="r"), {"text": "done"}])
    assert run.stopped == "answered" and run.answer == "done" and run.steps == 2
    assert (work / "r" / "summary.json").exists()
    events = al.read_trace(run.trace)
    assert [e["event"] for e in events] == ["task", "assistant", "tool", "assistant", "tool", "assistant", "end"]
    assert al.ScriptedLLM.from_trace(run.trace).script[0]["tool_calls"][0]["name"] == "validate"      # a trace can be replayed


def test_bad_calls_are_answered_not_run(work):
    run = run_script(work, [call("nothing"), call("run", model="model.json", surprise=1), call("read_file", path="../secret.txt"),
                            call("run", model="../model.json"), {"text": "x"}])
    assert [c["code"] for c in run.calls] == ["unknown_tool", "invalid_arguments", "bad_path", "bad_path"]
    assert not (work / "summary.json").exists()


def test_approval_gates(work):
    ensemble = call("ensemble", model="model.json", method="esmda", observations="o.json", parameters="p.json", prior="q.json")
    (work / "keep.md").write_text("mine")
    run = run_script(work, [ensemble, call("write_file", path="keep.md", content="theirs"),
                            call("write_file", path="report.md", content="conclusion"), call("write_file", path="new.txt", content="fine"),
                            {"text": "x"}])
    assert [c["code"] for c in run.calls] == ["declined", "declined", "declined", None]
    assert (work / "keep.md").read_text() == "mine" and not (work / "report.md").exists() and (work / "new.txt").exists()
    run = run_script(work, [call("write_file", path="report.md", content="ok"), {"text": "x"}], approve=al.auto_approve)
    assert (work / "report.md").read_text() == "ok"


def test_a_missing_optional_file_gets_a_useful_error(work):
    """What a hosted model did: it named a schedule.json that does not exist. The answer must say what to do about it."""
    run = run_script(work, [call("run", model="model.json", schedule="schedule.json", out="r"), {"text": "x"}])
    assert [c["code"] for c in run.calls] == ["file_not_found"]
    answer = json.loads([e for e in al.read_trace(run.trace) if e["event"] == "tool"][0]["result"])
    assert "model.json" in answer["files_nearby"] and "leave the option out" in answer["hint"]
    assert "Optional arguments" in al.SYSTEM_PROMPT and "never after a single failure" in al.SYSTEM_PROMPT


def test_only_the_first_of_several_calls_runs(work):
    both = {"tool_calls": [{"name": "list_files", "arguments": {}}, {"name": "run", "arguments": {"model": "model.json", "out": "r"}}]}
    run = run_script(work, [both, {"text": "x"}])
    assert [c["code"] for c in run.calls] == [None, "not_run"] and not (work / "r").exists()
    run = run_script(work, [both, {"text": "x"}], one_call_at_a_time=False, trace="t2.jsonl")
    assert (work / "r" / "summary.json").exists()


def test_repeated_call_and_step_limit(work):
    run = run_script(work, [call("list_files")] * 6, max_steps=5)
    assert run.stopped == "max_steps" and "step limit" in run.answer
    assert [c["code"] for c in run.calls][-3:] == ["repeated_call"] * 3 and run.calls[0]["ok"]


def test_http_transport(work):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    transport = al.HttpTransport.start(al.Workspace(work), port=8791)
    try:
        run = run_script(work, [call("validate", model="model.json"), call("validate", model="missing.json"), {"text": "x"}], transport=transport)
    finally:
        transport.close()
    assert run.calls[0]["ok"] is True and run.calls[1]["ok"] is False


# ---------------------------------------------------------------------- the three tasks, scripted
@pytest.mark.slow
def test_scripted_tasks(tmp_path):
    import agent_lab_case as case
    folders = case.make_workspace(tmp_path)
    work = folders["work"]
    kwargs = dict(approve=al.auto_approve, verbose=False, max_steps=15)

    def go(name):
        return al.Agent(al.connect("scripted", script=scripts.SCRIPTS[name]), work, trace=f"trace_{name}.jsonl", **kwargs).run(scripts.TASKS[name])

    a = go("A")
    assert a.stopped == "answered" and all(c["ok"] for c in a.calls) and "STB" in a.answer
    b = go("B")
    assert [c["code"] for c in b.calls if not c["ok"]] == ["invalid_arguments"] and "healthy = False" in b.answer
    assert "healthy = True" in b.answer
    c = go("C")
    errors = [x["code"] for x in c.calls if not x["ok"]]
    assert errors == ["unmapped_well"] and (work / "report.md").exists()
    events = al.read_trace(c.trace)
    misfits = [json.loads(e["result"])["rms"] for e in events if e.get("tool") == "misfit"]
    assert misfits[2] < 0.1 * misfits[0] and misfits[3] > 3 * misfits[2]            # fits the history, not the later data as well
    score = case.score_against_truth(work / "calibration" / "posterior_mean.json", folders["hidden"])
    assert score["calibrated"] < score["deck_model"]                                  # the data helped, but only partly: see the notebook
