"""An HTTP API for kusim (FastAPI): the command line over the network, for agents and services that have no shell.

    pip install kusim[api]
    kusim serve --workspace ./work            # then open http://127.0.0.1:8000/docs

It is a thin layer. Every command of the command line that runs or edits a model is a ``POST /v1/commands/{name}``
whose JSON body holds the same options as the command (``--max-cells`` is ``max_cells``), and the answer is the very
JSON object the command prints. Each command runs in its **own process** (``python -m kusim ...``), so a request cannot
corrupt another, a runaway run is killed at the timeout, and the contract is exactly the one of the command line
(``docs/json-format.md``, ``AGENTS.md``).

Files live in a **workspace** folder on the server. A client stores JSON documents there (``PUT /v1/files/model.json``),
names them in command bodies (``{"model": "model.json", "out": "runs/a"}``) and fetches results
(``GET /v1/files/runs/a/results.json``). Names are relative paths of simple segments; nothing outside the workspace can be
read or written.

Safety, because a network service is not a shell: the server has **ceilings** for the guardrail limits (a client may lower
``max_cells``, ``max_seconds``, ``max_output_mb`` and ``workers``, not raise them or switch them off), a wall-clock
**timeout** per command, a limit on the number of commands running at once (the next gets ``429``), a limit on the size of
a stored document, and an optional bearer **token**. It has no TLS and no user accounts: bind it to localhost, or put it
behind a reverse proxy, and do not expose it to the internet as it is.

Status codes: ``200`` for an answer with ``"ok": true`` (also a run that is not ``healthy``: read the body), ``202`` for a
background job, ``404`` for a missing file, ``422`` for bad input (the body has ``code``, ``path`` and the hints of the
command line), ``429`` when busy, ``401`` without a valid token, ``504`` at the timeout.
"""

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ConfigDict, Field, create_model

from . import __version__
from .checks import DEFAULT_MAX_CELLS, DEFAULT_MAX_OUTPUT_MB, DEFAULT_MAX_SECONDS
from .schema import SCHEMAS, get_schema
from .solver import METHODS
from .spec import FIELD_MODES

API_PREFIX = "/v1"
_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
UPLOAD_SUFFIXES = {".data", ".inc", ".csv", ".txt", ".xlsx", ".xlsm"}      # stored as they are (a deck, its includes, a workbook)


# ---------------------------------------------------------------------- configuration
@dataclass
class ServerConfig:
    """Everything the operator decides. ``max_cells``, ``max_seconds`` and ``max_output_mb`` are the ceilings of the
    guardrail limits of ``kusim.checks`` (a client may ask for less); ``timeout_seconds`` kills a command that runs
    longer; ``max_concurrent`` is the number of commands (or background jobs) at once."""

    workspace: Path = Path("kusim-workspace")
    token: Optional[str] = None
    timeout_seconds: float = 900.0
    max_concurrent: int = 2
    max_workers: int = 2
    max_cells: int = DEFAULT_MAX_CELLS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_output_mb: float = DEFAULT_MAX_OUTPUT_MB
    max_document_mb: float = 50.0
    job_ttl_seconds: float = 3600.0
    python: str = field(default_factory=lambda: sys.executable)


class ApiError(Exception):
    def __init__(self, status, code, message, **details):
        super().__init__(message)
        self.status, self.code, self.details = status, code, details


# ---------------------------------------------------------------------- the commands as a table
# (field, kind, flag, required): ``flag`` None means positional. kinds: in_file / out_file (workspace names), str,
# int, float, bool (a switch), list (a repeated option), or a tuple of choices. tests/test_api.py checks that this table
# and the argparse parser of the command line list the same options.
_GUARD = [("max_cells", "int", "--max-cells", False), ("max_seconds", "float", "--max-seconds", False)]
_GUARD_OUTPUT = _GUARD + [("max_output_mb", "float", "--max-output-mb", False)]
_UPSTREAM = ("potential", "pressure")

COMMANDS: Dict[str, List[tuple]] = {
    "template": [("output", "out_file", "-o", False), ("nx", "int", "--nx", False), ("ny", "int", "--ny", False),
                 ("nz", "int", "--nz", False), ("days", "float", "--days", False)],
    "validate": [("model", "in_file", None, True), ("schedule", "in_file", "--schedule", False),
                 ("restart", "in_file", "--restart", False), ("no_probe", "bool", "--no-probe", False),
                 ("fields", FIELD_MODES, "--fields", False)] + _GUARD_OUTPUT,
    "edit": [("model", "in_file", None, True), ("set", "list", "--set", False), ("patch", "in_file", "--patch", False),
             ("output", "out_file", "-o", False), ("in_place", "bool", "--in-place", False)],
    "diff": [("a", "in_file", None, True), ("b", "in_file", None, True)],
    "describe": [("model", "in_file", None, True)],
    "refine": [("model", "in_file", None, True), ("output", "out_file", "-o", True), ("factor", "int", "--factor", False),
               ("fx", "int", "--fx", False), ("fy", "int", "--fy", False), ("fz", "int", "--fz", False),
               ("dt", "str", "--dt", False), ("max_cells", "int", "--max-cells", False)],
    "compare": [("a", "in_file", None, True), ("b", "in_file", None, True)],
    "tables": [("file", "in_file", None, True)],
    "plot-template": [("name", "str", "--name", False), ("output", "out_file", "-o", False)],
    "plot": [("spec", "in_file", None, True), ("output", "out_file", "-o", True), ("width", "int", "--width", False),
             ("height", "int", "--height", False), ("scale", "float", "--scale", False)],
    "import-deck": [("deck", "in_file", None, True), ("output", "out_file", "-o", True),
                    ("schedule_out", "out_file", "--schedule-out", False), ("wells_out", "out_file", "--wells-out", False),
                    ("dt", "float", "--dt", False), ("pvt_pressures", "str", "--pvt-pressures", False),
                    ("skip_unknown", "bool", "--skip-unknown", False), ("no_probe", "bool", "--no-probe", False)],
    "import-history": [("input", "in_file", None, True), ("output", "out_file", "-o", True),
                       ("controls_out", "out_file", "--controls-out", False), ("model", "in_file", "--model", False),
                       ("sheet", "str", "--sheet", False), ("layout", ("auto", "by_well", "wide", "long"), "--layout", False),
                       ("header_row", "int", "--header-row", False), ("start_date", "str", "--start-date", False),
                       ("dayfirst", "bool", "--dayfirst", False), ("dt", "float", "--dt", False),
                       ("every", "float", "--every", False), ("align", ("mean", "last"), "--align", False),
                       ("well_map", "list", "--well-map", False), ("wells_file", "in_file", "--wells-file", False),
                       ("vectors", "str", "--vectors", False), ("pressure_unit", "str", "--pressure-unit", False),
                       ("rate_unit", "str", "--rate-unit", False), ("water_cut_unit", "str", "--water-cut-unit", False),
                       ("sigma", "list", "--sigma", False),
                       ("producer_control", ("liquid_rate", "oil_rate"), "--producer-control", False)],
    "history-template": [("output", "out_file", "-o", True)],
    "run": [("model", "in_file", None, True), ("schedule", "in_file", "--schedule", False),
            ("out", "out_file", "--out", False), ("fields", FIELD_MODES, "--fields", False),
            ("csv", "bool", "--csv", False), ("solver", tuple(METHODS), "--solver", False),
            ("upstream", _UPSTREAM, "--upstream", False), ("save_every", "int", "--save-every", False),
            ("strict", "bool", "--strict", False), ("restart", "in_file", "--restart", False),
            ("until", "float", "--until", False), ("state_out", "out_file", "--state-out", False)] + _GUARD_OUTPUT,
    "misfit": [("model", "in_file", None, True), ("observations", "in_file", "--observations", True),
               ("schedule", "in_file", "--schedule", False), ("restart", "in_file", "--restart", False),
               ("upstream", _UPSTREAM, "--upstream", False), ("out", "out_file", "--out", False),
               ("strict", "bool", "--strict", False)] + _GUARD,
    "gradient": [("model", "in_file", None, True), ("observations", "in_file", "--observations", True),
                 ("parameters", "in_file", "--parameters", False), ("controls", "in_file", "--controls", False),
                 ("schedule", "in_file", "--schedule", False), ("restart", "in_file", "--restart", False),
                 ("x", "in_file", "--x", False), ("method", ("adjoint", "fd"), "--method", False),
                 ("check", "int", "--check", False), ("upstream", _UPSTREAM, "--upstream", False),
                 ("out", "out_file", "--out", False)] + _GUARD,
    "ensemble": [("model", "in_file", None, True), ("method", ("esmda", "enkf"), "--method", True),
                 ("observations", "in_file", "--observations", True), ("parameters", "in_file", "--parameters", True),
                 ("prior", "in_file", "--prior", True), ("schedule", "in_file", "--schedule", False),
                 ("members", "int", "--members", False), ("seed", "int", "--seed", False),
                 ("passes", "int", "--passes", False), ("alphas", "str", "--alphas", False),
                 ("localise", "float", "--localise", False), ("workers", "int", "--workers", False),
                 ("out", "out_file", "--out", False), ("write_members", "bool", "--write-members", False),
                 ("write_ensemble", "bool", "--write-ensemble", False), ("strict", "bool", "--strict", False)] + _GUARD,
}


def safe_name(name) -> str:
    """A workspace-relative name: simple segments separated by ``/``, no ``..``, no drive or absolute path.
    Raises ``ApiError`` (422, ``bad_path``) otherwise."""
    bad = ApiError(422, "bad_path", f"{name!r} is not a valid workspace name: use relative names like 'runs/a/model.json' "
                   "(letters, digits, '.', '_' and '-' in each part, no '..')", path=str(name))
    if not isinstance(name, str) or not name or len(name) > 200:
        raise bad
    parts = name.split("/")
    if any(p in ("", ".", "..") or not _SEGMENT.match(p) for p in parts):
        raise bad
    return name


def resolve(config: ServerConfig, name) -> Path:
    """The absolute path of a workspace name, guaranteed to be inside the workspace (symbolic links included)."""
    root = config.workspace.resolve()
    target = (root / safe_name(name)).resolve()
    if root != target and root not in target.parents:
        raise ApiError(422, "bad_path", f"{name!r} leads outside the workspace", path=name)
    return target


def build_argv(config: ServerConfig, command: str, body: Dict[str, Any]) -> List[str]:
    """The command line for a request body (validated names, server ceilings applied). Raises ``ApiError``."""
    ceilings = {"max_cells": config.max_cells, "max_seconds": config.max_seconds, "max_output_mb": config.max_output_mb,
                "workers": config.max_workers}
    positional, options = [], []
    for name, kind, flag, _ in COMMANDS[command]:
        value = body.get(name)
        if name in ceilings:
            if value is None:
                value = ceilings[name] if name != "workers" else None
            elif value <= 0 or value > ceilings[name]:
                raise ApiError(422, "limit_exceeds_server_ceiling", f"{name} = {value} is not allowed: this server's "
                               f"ceiling is {ceilings[name]:g} (a client may lower it, not raise it or switch it off)",
                               path=name, ceiling=ceilings[name])
        if value is None:
            continue
        if kind in ("in_file", "out_file"):
            resolve(config, value)                                       # validates; the command gets the relative name
        if kind == "bool":
            if value:
                options.append(flag)
        elif kind == "list":
            for item in value:
                options += [flag, str(item)]
        elif flag is None:
            positional.append(str(value))
        else:
            options += [flag, str(value)]
    return [command, *positional, *options]


# ---------------------------------------------------------------------- running a command in its own process
def _kill_tree(process):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        else:
            import signal

            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


def run_process(config: ServerConfig, argv: List[str]):
    """Run ``python -m kusim argv`` in the workspace; returns ``(status, body)``."""
    env = dict(os.environ, PYTHONIOENCODING="utf8")
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    process = subprocess.Popen([config.python, "-m", "kusim", *argv], cwd=config.workspace, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env, text=True, encoding="utf8", **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        process.communicate()
        return 504, {"ok": False, "code": "timeout", "command": argv[0], "error":
                     f"the command ran longer than the server's timeout of {config.timeout_seconds:g} s and was stopped; "
                     "use a smaller model, a larger dt, fewer members, or run it from the command line"}
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError:
        return 500, {"ok": False, "code": "internal_error", "error": "the command did not print a JSON object",
                     "stderr": stderr[-2000:]}
    if process.returncode == 2 and isinstance(body, dict):
        return (404 if "not found" in str(body.get("error", "")) else 422), body
    if process.returncode not in (0, 1):
        return 500, {"ok": False, "code": "internal_error", "error": f"exit code {process.returncode}",
                     "stderr": stderr[-2000:], "answer": body}
    return 200, body


# ---------------------------------------------------------------------- background jobs
@dataclass
class Job:
    id: str
    command: str
    status: str = "running"                                 # running, done, failed, timeout
    http_status: Optional[int] = None
    answer: Any = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None

    def public(self):
        return {"id": self.id, "command": self.command, "status": self.status, "http_status": self.http_status,
                "seconds": round((self.finished or time.time()) - self.started, 3), "answer": self.answer}


class Jobs:
    def __init__(self, config: ServerConfig):
        self.config, self.items, self.lock = config, {}, threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max(config.max_concurrent, 1))

    def _expire(self):
        cutoff = time.time() - self.config.job_ttl_seconds
        for key in [k for k, j in self.items.items() if j.finished and j.finished < cutoff]:
            del self.items[key]

    def submit(self, command, argv, release):
        job = Job(uuid.uuid4().hex[:12], command)
        with self.lock:
            self._expire()
            self.items[job.id] = job

        def work():
            try:
                job.http_status, job.answer = run_process(self.config, argv)
                job.status = "timeout" if job.http_status == 504 else ("done" if job.http_status in (200,) else "failed")
            except Exception as error:                                           # never leave a job "running" forever
                job.http_status, job.answer, job.status = 500, {"ok": False, "code": "internal_error", "error": str(error)}, "failed"
            finally:
                job.finished = time.time()
                release()

        self.pool.submit(work)
        return job

    def get(self, job_id):
        with self.lock:
            self._expire()
            return self.items.get(job_id)

    def all(self):
        with self.lock:
            self._expire()
            return list(self.items.values())


# ---------------------------------------------------------------------- the application
def _request_model(command, spec):
    fields = {}
    for name, kind, flag, required in spec:
        if kind in ("in_file", "out_file"):
            kind_type, text = str, ("a workspace file name" + (" to write" if kind == "out_file" else " to read"))
        elif kind == "bool":
            kind_type, text = bool, "a switch"
        elif kind == "list":
            kind_type, text = List[str], "a list: each item is one repetition of the option"
        elif isinstance(kind, tuple):
            kind_type, text = Literal[kind], "one of " + ", ".join(kind)
        else:
            kind_type, text = {"int": int, "float": float, "str": str}[kind], kind
        description = f"{'the argument' if flag is None else 'the option ' + flag} ({text})"
        fields[name] = (kind_type, Field(..., description=description)) if required else \
                       (Optional[kind_type], Field(None, description=description))
    return create_model(f"{command.replace('-', '_').capitalize()}Request", __config__=ConfigDict(extra="forbid"), **fields)


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    config = config or ServerConfig()
    config.workspace = Path(config.workspace)
    config.workspace.mkdir(parents=True, exist_ok=True)
    slots = threading.BoundedSemaphore(max(config.max_concurrent, 1))
    jobs = Jobs(config)
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(title="kusim", version=__version__, description=__doc__)

    def authorise(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        if config.token and not (credentials and secrets.compare_digest(credentials.credentials, config.token)):
            raise ApiError(401, "unauthorized", "a valid bearer token is required")

    @app.exception_handler(ApiError)
    async def api_error(request, error):
        return JSONResponse({"ok": False, "error": str(error), "code": error.code, **error.details}, error.status,
                            headers={"WWW-Authenticate": "Bearer"} if error.status == 401 else None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        first = error.errors()[0]
        path = ".".join(str(p) for p in first["loc"][1:])
        code = {"extra_forbidden": "unknown_key", "missing": "missing_key"}.get(first["type"], "bad_value")
        return JSONResponse({"ok": False, "error": f"{path or 'the request'}: {first['msg']}", "code": code, "path": path,
                             "details": json.loads(json.dumps(error.errors(), default=str))}, 422)

    # ---- information
    @app.get(f"{API_PREFIX}/health", tags=["info"])
    def health():
        return {"ok": True, "kusim": __version__}

    @app.get(f"{API_PREFIX}/schema/{{name}}", tags=["info"], dependencies=[Depends(authorise)])
    def schema(name: str):
        if name not in SCHEMAS:
            raise ApiError(404, "unknown_schema", f"no schema {name!r}; choose one of {sorted(SCHEMAS)}", allowed=sorted(SCHEMAS))
        return get_schema(name)

    @app.get(f"{API_PREFIX}/limits", tags=["info"], dependencies=[Depends(authorise)])
    def limits():
        return {"ok": True, "timeout_seconds": config.timeout_seconds, "max_concurrent": config.max_concurrent,
                "max_workers": config.max_workers, "max_cells": config.max_cells, "max_seconds": config.max_seconds,
                "max_output_mb": config.max_output_mb, "max_document_mb": config.max_document_mb}

    # ---- the workspace
    @app.put(API_PREFIX + "/files/{name:path}", tags=["files"], dependencies=[Depends(authorise)])
    async def put_file(name: str, request: Request):
        target = resolve(config, name)
        raw = await request.body()
        if len(raw) > config.max_document_mb * 1e6:
            raise ApiError(413, "too_large", f"a document may be at most {config.max_document_mb:g} MB")
        if Path(name).suffix.lower() not in UPLOAD_SUFFIXES:
            try:
                json.loads(raw)
            except json.JSONDecodeError as error:
                raise ApiError(422, "invalid_json", f"the body is not valid JSON: {error} (documents other than JSON must "
                               f"have one of the extensions {sorted(UPLOAD_SUFFIXES)})", line=error.lineno, column=error.colno)
        if target.is_dir():
            raise ApiError(422, "bad_path", f"{name!r} is a folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {"ok": True, "path": name, "bytes": len(raw)}

    @app.get(API_PREFIX + "/files/{name:path}", tags=["files"], dependencies=[Depends(authorise)])
    def get_file(name: str):
        target = resolve(config, name)
        if not target.is_file():
            raise ApiError(404, "file_not_found", f"file not found: {name}", path=name)
        media = {".json": "application/json", ".png": "image/png"}.get(target.suffix.lower(), "application/octet-stream")
        return FileResponse(target, media_type=media)

    @app.delete(API_PREFIX + "/files/{name:path}", tags=["files"], dependencies=[Depends(authorise)])
    def delete_file(name: str):
        target = resolve(config, name)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
        else:
            raise ApiError(404, "file_not_found", f"file not found: {name}", path=name)
        return {"ok": True, "deleted": name}

    @app.get(f"{API_PREFIX}/files", tags=["files"], dependencies=[Depends(authorise)])
    def list_files(prefix: str = Query("", description="only names starting with this")):
        root = config.workspace.resolve()
        found = sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file())
        return {"ok": True, "files": [{"path": n, "bytes": s} for n, s in found if n.startswith(prefix)][:1000]}

    # ---- the commands
    def endpoint_for(command, Model):
        def endpoint(body: Model, background: bool = Query(False, alias="async",
                                                           description="return 202 and a job id; poll GET /v1/jobs/{id}")):
            argv = build_argv(config, command, body.model_dump(exclude_none=True))
            if not slots.acquire(blocking=False):
                raise ApiError(429, "busy", f"{config.max_concurrent} commands are running: try again shortly",
                               retry_after_seconds=5)
            if background:
                job = jobs.submit(command, argv, slots.release)
                return JSONResponse({"ok": True, "job": job.id, "status": job.status, "poll": f"{API_PREFIX}/jobs/{job.id}"}, 202)
            try:
                status, answer = run_process(config, argv)
            finally:
                slots.release()
            return JSONResponse(answer, status)

        endpoint.__name__ = f"command_{command}"
        endpoint.__doc__ = f"``kusim {command}``: the body holds the options of the command (see ``kusim {command} --help``)."
        return endpoint

    for command, spec in COMMANDS.items():
        app.post(f"{API_PREFIX}/commands/{command}", tags=["commands"], dependencies=[Depends(authorise)])(
            endpoint_for(command, _request_model(command, spec)))

    @app.get(API_PREFIX + "/jobs/{job_id}", tags=["jobs"], dependencies=[Depends(authorise)])
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise ApiError(404, "unknown_job", f"no job {job_id!r} (jobs are forgotten {config.job_ttl_seconds:g} s after they end)")
        return job.public()

    @app.get(f"{API_PREFIX}/jobs", tags=["jobs"], dependencies=[Depends(authorise)])
    def list_jobs():
        return {"ok": True, "jobs": [{k: v for k, v in j.public().items() if k != "answer"} for j in jobs.all()]}

    return app
