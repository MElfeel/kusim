"""The HTTP API (kusim.api): the commands over the network, the workspace, and the safety limits."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from kusim.api import API_PREFIX, COMMANDS, ApiError, ServerConfig, build_argv, create_app, resolve, safe_name  # noqa: E402
from kusim.cli import build_parser  # noqa: E402

V1 = API_PREFIX


@pytest.fixture
def config(tmp_path):
    return ServerConfig(workspace=tmp_path / "work", timeout_seconds=120.0)


@pytest.fixture
def client(config):
    return TestClient(create_app(config))


def _put(client, name, data):
    response = client.put(f"{V1}/files/{name}", content=json.dumps(data))
    assert response.status_code == 200, response.text
    return response.json()


def _command(client, name, **body):
    return client.post(f"{V1}/commands/{name}", json=body)


def _build_twin(client):
    """A small model, observations of a truth with twice the permeability, a parameter set and a prior."""
    assert _command(client, "template", output="model.json", nx=4, ny=3, nz=2, days=30).status_code == 200
    model = client.get(f"{V1}/files/model.json").json()
    double = lambda key: [2.0 * v for v in model[key]] if isinstance(model[key], list) else 2.0 * model[key]
    edit = _command(client, "edit", model="model.json", output="truth.json",
                    set=[f"kx={json.dumps(double('kx'))}", f"ky={json.dumps(double('ky'))}", f"kz={json.dumps(double('kz'))}"])
    assert edit.status_code == 200, edit.text
    assert _command(client, "run", model="truth.json", out="truth", fields="none").status_code == 200
    rows = {(r["well"], r["time"]): r for r in client.get(f"{V1}/files/truth/results.json").json()["wells"]}
    observations = [{"time": t, "vector": v, "value": rows[(w, t)][c], "sigma": s}
                    for t in (10.0, 20.0, 30.0)
                    for v, (w, c, s) in {"WBHP:I1": ("I1", "bhp", 5.0), "WWCT:P1": ("P1", "water_cut", 0.02),
                                         "WOPR:P1": ("P1", "oil_rate", 20.0)}.items()]
    _put(client, "obs.json", observations)
    _put(client, "params.json", {"parameters": [{"type": "field", "field": "kx", "ties": ["ky", "kz"], "lower": 1.0}]})
    _put(client, "prior.json", {"parameters": {"log_kx": {"std": 0.6, "range_xy": 150.0, "range_z": 20.0}}})


@pytest.fixture(scope="module")
def twin_dir(tmp_path_factory):
    """The twin built once (it takes a few commands, each a process); tests get a copy of the folder."""
    folder = tmp_path_factory.mktemp("twin") / "work"
    _build_twin(TestClient(create_app(ServerConfig(workspace=folder, timeout_seconds=120.0))))
    return folder


@pytest.fixture
def twin(client, config, twin_dir):
    shutil.copytree(twin_dir, config.workspace, dirs_exist_ok=True)


# ---------------------------------------------------------------------- the table matches the command line
def _cli_options(name):
    sub = build_parser()._subparsers._group_actions[0].choices[name]
    flags, positionals = [], []
    for action in sub._actions:
        if action.dest == "help":
            continue
        if action.option_strings:
            flags.append(set(action.option_strings))              # an option may have a short and a long name
        else:
            positionals.append(action.dest)
    return flags, positionals


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_option_of_the_command_line_is_in_the_table(command):
    """If a command gains an option, the API must get it too (or this test says so)."""
    flags, positionals = _cli_options(command)
    table_flags = [flag for _, _, flag, _ in COMMANDS[command] if flag]
    table_positionals = [name for name, _, flag, _ in COMMANDS[command] if flag is None]
    assert len(table_flags) == len(flags) and all(any(f in names for f in table_flags) for names in flags), command
    assert all(sum(f in names for names in flags) == 1 for f in table_flags), command
    assert table_positionals == positionals, command


def test_commands_not_in_the_api_are_the_ones_that_make_no_sense_there():
    parser_commands = set(build_parser()._subparsers._group_actions[0].choices)
    assert parser_commands - set(COMMANDS) == {"version", "convert", "schema", "serve"}


# ---------------------------------------------------------------------- names and paths
@pytest.mark.parametrize("name", ["model.json", "runs/a/results.json", "a-b_c.1/x", "x" * 100])
def test_good_names(name):
    assert safe_name(name) == name


@pytest.mark.parametrize("name", ["", "/etc/passwd", "../x", "a/../b", "a//b", "a/./b", "C:/x", "a\\b", "a b", ".", "..",
                                  "x" * 201, "a/" + "y" * 101, None, 5])
def test_bad_names(name):
    with pytest.raises(ApiError) as error:
        safe_name(name)
    assert error.value.status == 422 and error.value.code == "bad_path"


def test_a_link_out_of_the_workspace_is_refused(config):
    config.workspace.mkdir(parents=True)
    outside = config.workspace.parent / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, config.workspace / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symbolic links here")
    with pytest.raises(ApiError, match="outside the workspace"):
        resolve(config, "link/secret.json")


def test_argv_follows_the_command_line(config):
    argv = build_argv(config, "ensemble", {"model": "m.json", "method": "esmda", "observations": "o.json",
                                           "parameters": "p.json", "prior": "q.json", "members": 12, "write_members": True,
                                           "strict": False})
    assert argv[:2] == ["ensemble", "m.json"] and "--write-members" in argv and "--strict" not in argv
    assert argv[argv.index("--members") + 1] == "12"
    assert argv[argv.index("--max-cells") + 1] == str(config.max_cells)          # the server's ceiling is passed on
    edit = build_argv(config, "edit", {"model": "m.json", "set": ["kx=1", "dt=2"], "in_place": True})
    assert edit == ["edit", "m.json", "--set", "kx=1", "--set", "dt=2", "--in-place"]


# ---------------------------------------------------------------------- files
def test_files_round_trip_list_and_delete(client):
    assert _put(client, "a/b/doc.json", {"x": [1, 2]}) == {"ok": True, "path": "a/b/doc.json",
                                                            "bytes": len(json.dumps({"x": [1, 2]}))}
    assert client.get(f"{V1}/files/a/b/doc.json").json() == {"x": [1, 2]}
    assert [f["path"] for f in client.get(f"{V1}/files", params={"prefix": "a/"}).json()["files"]] == ["a/b/doc.json"]
    assert client.delete(f"{V1}/files/a").json() == {"ok": True, "deleted": "a"}                # a folder too
    assert client.get(f"{V1}/files/a/b/doc.json").status_code == 404
    assert client.delete(f"{V1}/files/a").status_code == 404


def test_file_errors(client, config):
    bad = client.put(f"{V1}/files/x.json", content="{not json")
    assert bad.status_code == 422 and bad.json()["code"] == "invalid_json" and "line" in bad.json()
    missing = client.get(f"{V1}/files/nope.json")
    assert missing.status_code == 404 and missing.json()["code"] == "file_not_found"
    assert client.get(f"{V1}/files/..%2Fsecret").status_code in (404, 422)
    config.max_document_mb = 0.001
    big = client.put(f"{V1}/files/big.json", content=json.dumps(list(range(2000))))
    assert big.status_code == 413 and big.json()["code"] == "too_large"
    client.put(f"{V1}/files/d/doc.json", content="1")
    assert client.put(f"{V1}/files/d", content="1").status_code == 422                        # a folder is not a document


def test_the_schema_endpoint(client):
    schema = client.get(f"{V1}/schema/model").json()
    assert schema["title"] == "kusim model"
    unknown = client.get(f"{V1}/schema/nothing")
    assert unknown.status_code == 404 and "model" in unknown.json()["allowed"]


# ---------------------------------------------------------------------- the commands
def test_an_answer_is_the_answer_of_the_command_line(client, config, capsys):
    from kusim.cli import main

    _command(client, "template", output="model.json", nx=4, ny=3, nz=2, days=10)
    api = _command(client, "run", model="model.json").json()
    code = main(["run", str(config.workspace / "model.json")])
    cli = json.loads(capsys.readouterr().out)
    assert code == 0 and api["ok"]
    for key in ("final_time", "n_steps", "wells", "diagnostics", "healthy", "stop_reason"):
        assert api[key] == cli[key], key


@pytest.mark.slow
def test_a_whole_session_over_http(client, twin):
    """The agent session of tests/test_agent_workflow.py, through the API."""
    misfit = _command(client, "misfit", model="model.json", observations="obs.json")
    assert misfit.status_code == 200 and misfit.json()["rms"] > 2.0
    matched = _command(client, "ensemble", model="model.json", method="esmda", observations="obs.json",
                       parameters="params.json", prior="prior.json", members=12, seed=3, out="match")
    assert matched.status_code == 200, matched.text
    answer = matched.json()
    assert answer["healthy"] and answer["metrics"]["posterior_member_rms"] < 0.5 * answer["metrics"]["prior_member_rms"]
    assert answer["written"]["posterior_mean_model"] == os.path.join("match", "posterior_mean.json") or \
        answer["written"]["posterior_mean_model"].replace("\\", "/") == "match/posterior_mean.json"
    after = _command(client, "misfit", model="match/posterior_mean.json", observations="obs.json").json()
    assert after["rms"] < 0.5 * misfit.json()["rms"]
    diff = _command(client, "diff", a="model.json", b="match/posterior_mean.json").json()
    assert not diff["identical"]
    gradient = _command(client, "gradient", model="match/posterior_mean.json", observations="obs.json",
                        parameters="params.json", check=2)
    assert gradient.status_code == 200 and gradient.json()["check"]["max_relative_error"] < 1e-3
    first = _command(client, "run", model="model.json", until=15, state_out="s15.json").json()
    second = _command(client, "run", model="model.json", restart="s15.json", state_out="s30.json").json()
    whole = _command(client, "run", model="model.json", state_out="whole.json")
    assert first["final_time"] == 15.0 and second["start_time"] == 15.0 and whole.status_code == 200
    assert client.get(f"{V1}/files/s30.json").json() == client.get(f"{V1}/files/whole.json").json()


def test_errors_of_the_command_line_arrive_with_their_hints(client, twin):
    typo = _command(client, "edit", model="model.json", output="n.json", set=["producers.0.min_bph=1"])
    assert typo.status_code == 422 and typo.json()["code"] == "unknown_key"
    assert typo.json()["did_you_mean"] == {"min_bph": "min_bhp"}
    missing = _command(client, "run", model="nothing.json")
    assert missing.status_code == 404 and missing.json()["ok"] is False
    unhealthy = _command(client, "misfit", model="model.json", observations="obs.json", max_cells=1)
    assert unhealthy.status_code == 422 and unhealthy.json()["code"] == "too_large"


def test_error_bodies_do_not_leak_the_server_path(client, config):
    text = _command(client, "run", model="nothing.json").text + _command(client, "validate", model="x/y.json").text
    assert str(config.workspace) not in text and str(config.workspace.resolve()) not in text


def test_bad_request_bodies(client):
    unknown = _command(client, "run", model="m.json", bogus=1)
    assert unknown.status_code == 422 and unknown.json()["code"] == "unknown_key" and unknown.json()["path"] == "bogus"
    missing = client.post(f"{V1}/commands/run", json={})
    assert missing.json()["code"] == "missing_key" and missing.json()["path"] == "model"
    wrong = _command(client, "run", model="m.json", solver="magic")
    assert wrong.status_code == 422 and wrong.json()["path"] == "solver"
    traversal = _command(client, "run", model="../../etc/passwd")
    assert traversal.status_code == 422 and traversal.json()["code"] == "bad_path"
    assert _command(client, "run", model="m.json", out="/tmp/x").json()["code"] == "bad_path"


def test_clients_may_lower_the_limits_but_not_raise_them(client, config, twin):
    config.max_cells = 100
    raised = _command(client, "run", model="model.json", max_cells=1000)
    assert raised.status_code == 422 and raised.json()["code"] == "limit_exceeds_server_ceiling"
    assert raised.json()["ceiling"] == 100 and raised.json()["path"] == "max_cells"
    off = _command(client, "run", model="model.json", max_seconds=0)
    assert off.status_code == 422 and off.json()["code"] == "limit_exceeds_server_ceiling"
    assert _command(client, "run", model="model.json", max_cells=50).status_code == 200      # lowering is fine
    config.max_workers = 1
    workers = _command(client, "ensemble", model="model.json", method="esmda", observations="obs.json",
                       parameters="params.json", prior="prior.json", workers=4)
    assert workers.status_code == 422 and workers.json()["path"] == "workers"


def test_limits_endpoint(client, config):
    body = client.get(f"{V1}/limits").json()
    assert body["max_cells"] == config.max_cells and body["timeout_seconds"] == config.timeout_seconds


def test_a_command_over_the_timeout_is_stopped(config):
    config.timeout_seconds = 0.05
    client = TestClient(create_app(config))
    (config.workspace / "m.json").write_text("{}")
    answer = client.post(f"{V1}/commands/validate", json={"model": "m.json"})
    assert answer.status_code == 504 and answer.json()["code"] == "timeout"


# ---------------------------------------------------------------------- jobs, concurrency
def _wait(client, job_id, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"{V1}/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("the job did not finish")


@pytest.mark.slow
def test_a_background_job_and_the_busy_signal(config):
    config.max_concurrent = 1
    client = TestClient(create_app(config))
    _command(client, "template", output="model.json", nx=4, ny=3, nz=2, days=30)
    model = client.get(f"{V1}/files/model.json").json()
    _put(client, "obs.json", [{"time": 10.0, "vector": "WBHP:I1", "value": 3000.0, "sigma": 20.0},
                              {"time": 30.0, "vector": "WWCT:P1", "value": 0.1, "sigma": 0.05}])
    _put(client, "params.json", {"parameters": [{"type": "field", "field": "kx", "ties": ["ky", "kz"]}]})
    _put(client, "prior.json", {"parameters": {"log_kx": {"std": 0.5}}})
    started = client.post(f"{V1}/commands/ensemble", params={"async": "true"},
                          json={"model": "model.json", "method": "esmda", "observations": "obs.json",
                                "parameters": "params.json", "prior": "prior.json", "members": 20, "passes": 4})
    assert started.status_code == 202 and started.json()["status"] == "running"
    busy = _command(client, "validate", model="model.json")
    assert busy.status_code == 429 and busy.json()["code"] == "busy"
    job = _wait(client, started.json()["job"])
    assert job["status"] == "done" and job["answer"]["ok"] and job["http_status"] == 200
    assert _command(client, "validate", model="model.json").status_code == 200               # the slot is free again
    listed = client.get(f"{V1}/jobs").json()["jobs"]
    assert [j["id"] for j in listed] == [started.json()["job"]] and "answer" not in listed[0]
    assert client.get(f"{V1}/jobs/nope").status_code == 404
    assert model["dt"] > 0


def test_a_failed_job_keeps_the_error(client):
    started = client.post(f"{V1}/commands/run", params={"async": "true"}, json={"model": "nothing.json"})
    job = _wait(client, started.json()["job"])
    assert job["status"] == "failed" and job["http_status"] == 404 and job["answer"]["ok"] is False


# ---------------------------------------------------------------------- authentication
def test_bearer_token(config):
    config.token = "s3cret"
    client = TestClient(create_app(config))
    assert client.get(f"{V1}/health").status_code == 200                      # liveness is open
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic abc"}):
        refused = client.get(f"{V1}/limits", headers=headers)
        assert refused.status_code == 401 and refused.json()["code"] == "unauthorized"
    assert client.get(f"{V1}/limits", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.post(f"{V1}/commands/template", json={}).status_code == 401
    assert client.put(f"{V1}/files/x.json", content="1").status_code == 401


# ---------------------------------------------------------------------- the real server
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_kusim_serve_starts_a_real_server(tmp_path):
    import httpx

    port = _free_port()
    process = subprocess.Popen([sys.executable, "-m", "kusim", "serve", "--workspace", str(tmp_path / "w"), "--port", str(port),
                                "--token", "t0k"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lines = []
        while not lines or lines[-1].strip() != "}":                       # the first answer is one indented JSON object
            lines.append(process.stdout.readline())
        first = json.loads("".join(lines))
        assert first["ok"] and first["token_required"] and first["serving"].endswith(f":{port}/v1")
        base, headers = f"http://127.0.0.1:{port}{V1}", {"Authorization": "Bearer t0k"}
        for _ in range(100):
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        assert httpx.get(f"{base}/health").json()["ok"] is True
        assert httpx.post(f"{base}/commands/template", json={"output": "m.json"}).status_code == 401
        made = httpx.post(f"{base}/commands/template", json={"output": "m.json", "nx": 3, "ny": 2, "nz": 1, "days": 5},
                          headers=headers, timeout=60)
        assert made.status_code == 200 and made.json()["n_cells"] == 6
        ran = httpx.post(f"{base}/commands/run", json={"model": "m.json"}, headers=headers, timeout=60)
        assert ran.status_code == 200 and ran.json()["final_time"] == 5.0
    finally:
        process.terminate()
        process.wait(timeout=20)


# ---------------------------------------------------------------------- decks and workbooks over HTTP
def test_a_deck_and_a_workbook_are_uploaded_as_they_are_and_imported(client, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / "examples" / "eclipse"
    for name in ("waterflood.DATA", "waterflood_props.INC"):
        response = client.put(f"{V1}/files/deck/{name}", content=(folder / name).read_bytes())
        assert response.status_code == 200, response.text
    answer = _command(client, "import-deck", deck="deck/waterflood.DATA", output="model.json", schedule_out="schedule.json",
                      wells_out="wells.json")
    assert answer.status_code == 200 and answer.json()["wells"] == {"I1": "I1", "P1": "P1", "P2": "P2"}, answer.text
    assert client.get(f"{V1}/files/model.json").json()["initial"]["datum_depth"] == 7600.0

    book = openpyxl.Workbook()
    book.active.append(["Days", "Well", "BHP (bar)", "Water cut (%)"])
    book.active.append([10, "P1", 100.0, 2.0])
    path = tmp_path / "h.xlsx"
    book.save(path)
    assert client.put(f"{V1}/files/data/h.xlsx", content=path.read_bytes()).status_code == 200
    answer = _command(client, "import-history", input="data/h.xlsx", output="obs.json", model="model.json",
                      sigma=["WBHP=20", "WWCT=0.02"], vectors="WBHP,WWCT")
    assert answer.status_code == 200, answer.text
    rows = client.get(f"{V1}/files/obs.json").json()
    assert {r["vector"] for r in rows} == {"WBHP:P1", "WWCT:P1"} and rows[0]["value"] in (pytest.approx(1450.377), pytest.approx(0.02))
    assert _command(client, "misfit", model="model.json", observations="obs.json").status_code == 200

    template = _command(client, "history-template", output="template.xlsx")
    assert template.status_code == 200 and client.get(f"{V1}/files/template.xlsx").content[:2] == b"PK"      # an xlsx is a zip


def test_json_names_are_still_checked_but_other_documents_are_not(client):
    assert client.put(f"{V1}/files/x.json", content=b"not json").status_code == 422
    assert client.put(f"{V1}/files/x.DATA", content=b"RUNSPEC\n").status_code == 200


def test_import_errors_arrive_with_their_location(client):
    assert client.put(f"{V1}/files/bad.DATA", content=b"RUNSPEC\nPVTO\n 1 /\n/\n").status_code == 200
    answer = _command(client, "import-deck", deck="bad.DATA", output="m.json")
    assert answer.status_code == 422 and answer.json()["code"] == "unsupported_feature" and answer.json()["path"] == "bad.DATA:2"
