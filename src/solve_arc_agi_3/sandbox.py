"""Run model-written world-model code in a child process (ADR-0010, card ARC-51).

`SandboxedModel` implements the `WorldModel` protocol of `world_model.py`. The model source defines
`def predict(frame, move)`: `frame` is a list of hex-digit strings (one per row, not run-length encoded,
64 rows of 64 in practice), `move` is a dict such as `{"name": "ACTION3"}` or
`{"name": "ACTION6", "x": 12, "y": 30}`, and the result is a frame of the same shape.

This is **best-effort isolation, not a security boundary against a determined attacker**. What it is:
restricted builtins, an AST filter and an import allow-list inside the child, plus OS resource limits
(`resource`) and a wall-clock timeout enforced by the parent. It protects the agent from buggy and
lazy-hostile model code (infinite loops, memory bombs, stray file writes, `import os`). It does not stop
a skilled escape through the Python object graph, and it does not stop the child reading files the agent
user can read (no namespace or seccomp sandbox is used). That is acceptable here because the agent runs
offline on Kaggle and the child gets an empty environment, so there is no network and no secret in it.

Process model: one long-lived worker per `SandboxedModel`, started as `python -I -c <worker>` with an
empty environment and an empty temporary working directory, spoken to with one JSON object per line on
stdin and stdout. The state of the model's module globals persists between calls within a worker and is
lost when the worker is replaced. Nothing ever raises into the caller: failures are typed
`SandboxError`s, `predict` records the last one in `last_error` and returns an empty grid that cannot
match any real frame.
"""

from __future__ import annotations

import ast
import atexit
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, NoReturn, Self

from solve_arc_agi_3.world_model import Grid, Move, frame_to_rows, rows_to_frame

ErrorKind = Literal["timeout", "crash", "exception", "bad_output", "forbidden"]

HARD_TIMEOUT = 30.0
DEFAULT_TIMEOUT = 2.0
MAX_MESSAGE = 300
MAX_REPLY_BYTES = 1 << 20
ALLOWED_MODULES = (
    "math",
    "itertools",
    "collections",
    "functools",
    "copy",
    "heapq",
    "bisect",
    "re",
)


@dataclass(frozen=True)
class SandboxError:
    kind: ErrorKind
    message: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}" if self.message else self.kind


class SourceRejected(ValueError):
    """`load` refused the source before starting a worker (syntax error or no `predict`)."""

    def __init__(self, error: SandboxError) -> None:
        super().__init__(str(error))
        self.error = error


# Worker (runs in the child) -------------------------------------------------------------------------

_WORKER = r"""
import ast, json, os, resource, sys, types

MAX_PRINT = 65536
MAX_MOVES = 256
FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "input", "breakpoint", "exit", "quit", "help", "memoryview", "type", "dir", "id",
}
FORBIDDEN_ATTR_PREFIXES = ("_", "gi_", "f_", "co_", "cr_", "ag_", "tb_")
FORBIDDEN_ATTRS = {"mro", "format", "format_map"}
HEX = "0123456789abcdef"


class Forbidden(BaseException):
    pass


# Model code never sees a real module: an allowed module is wrapped in a proxy that carries only these
# names. Real modules leak others (re.enum.sys.modules reaches os), so no attribute that is itself a
# module is ever exposed either. None means every public non-module attribute.
EXPORTS = {
    "math": None, "itertools": None, "heapq": None, "bisect": None,
    "collections": {"deque", "Counter", "defaultdict", "OrderedDict", "namedtuple"},
    "functools": {"reduce", "lru_cache", "cache", "partial", "cmp_to_key"},
    "copy": {"copy", "deepcopy"},
    "re": {"compile", "match", "fullmatch", "search", "sub", "subn", "findall", "finditer", "split",
           "escape", "I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL", "X", "VERBOSE", "A", "ASCII"},
}


class SafeModule:
    pass


def make_proxy(real, names):
    proxy = SafeModule()
    for attr in dir(real):
        if attr.startswith("_"):
            continue
        value = getattr(real, attr)
        if isinstance(value, types.ModuleType):
            continue
        if names is None or attr in names:
            setattr(proxy, attr, value)
    return proxy


class TooMuchOutput(BaseException):
    pass


def check_ast(tree, allowed):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed or "." in a.name:
                    raise Forbidden("import of " + a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None or node.module.split(".")[0] not in allowed or "." in node.module:
                raise Forbidden("import from " + str(node.module))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise Forbidden("use of " + node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(FORBIDDEN_ATTR_PREFIXES) or node.attr in FORBIDDEN_ATTRS:
                raise Forbidden("attribute ." + node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise Forbidden("dunder string constant")


def main():
    import builtins

    rin = sys.stdin.buffer
    rout = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    json_dumps, json_loads = json.dumps, json.loads

    def send(obj):
        rout.write(json_dumps(obj).encode() + b"\n")

    init = json_loads(rin.readline())
    lim = init["limits"]
    for res, value in (
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_NPROC, lim["nproc"]),
        (resource.RLIMIT_NOFILE, lim["nofile"]),
        (resource.RLIMIT_CPU, lim["cpu"]),
        (resource.RLIMIT_AS, lim["memory"]),
    ):
        resource.setrlimit(res, (value, value))
    allowed = frozenset(init["allowed"])
    real_import = builtins.__import__

    printed = [0]

    def safe_print(*args, **kwargs):
        printed[0] += sum(len(str(a)) + 1 for a in args)
        if printed[0] > MAX_PRINT:
            raise TooMuchOutput()

    proxies = {}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in allowed or "." in name or name not in EXPORTS:
            raise Forbidden("import of " + name)
        if name not in proxies:
            proxies[name] = make_proxy(real_import(name), EXPORTS[name])
        return proxies[name]

    names = (
        "abs all any bool bytes bytearray chr dict divmod enumerate filter float frozenset hash hex bin oct int "
        "isinstance issubclass iter len list map max min next object ord pow property range repr reversed "
        "round set slice sorted staticmethod classmethod str sum super tuple zip callable "
        "Exception ValueError TypeError IndexError KeyError ZeroDivisionError StopIteration AssertionError "
        "ArithmeticError RuntimeError NotImplementedError OverflowError LookupError AttributeError"
    ).split()
    safe = {n: getattr(builtins, n) for n in names}
    safe["print"] = safe_print
    safe["__import__"] = safe_import
    safe["__build_class__"] = builtins.__build_class__
    glb = {"__builtins__": safe, "__name__": "world_model"}

    def run_init():
        try:
            tree = ast.parse(init["source"])
            check_ast(tree, allowed)
            exec(compile(tree, "<model>", "exec"), glb)
            if not callable(glb.get("predict")):
                send({"ok": False, "kind": "exception", "message": "source does not define predict"})
                return False
        except Forbidden as e:
            send({"ok": False, "kind": "forbidden", "message": str(e)})
            return False
        except SyntaxError as e:
            send({"ok": False, "kind": "exception", "message": "SyntaxError: " + str(e)[:200]})
            return False
        except MemoryError:
            send({"ok": False, "kind": "crash", "message": "MemoryError"})
            return False
        except Exception as e:
            send({"ok": False, "kind": "exception", "message": (type(e).__name__ + ": " + str(e))[:300]})
            return False
        send({"ok": True})
        return True

    if not run_init():
        return

    def valid(out, rows, width):
        if not isinstance(out, (list, tuple)) or len(out) != rows:
            return False
        return all(isinstance(r, str) and len(r) == width and all(c in HEX for c in r) for r in out)

    def valid_moves(out):
        if not isinstance(out, (list, tuple)) or len(out) > MAX_MOVES:
            return False
        for m in out:
            if not isinstance(m, dict) or not isinstance(m.get("name"), str):
                return False
            for k in ("x", "y"):
                if k in m and (type(m[k]) is not int):
                    return False
            if set(m) - {"name", "x", "y"}:
                return False
        return True

    predict = glb["predict"]
    while True:
        line = rin.readline()
        if not line:
            return
        req = json_loads(line)
        if req["op"] == "limits":
            send({"ok": True, "limits": {k: list(resource.getrlimit(getattr(resource, k)))
                  for k in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC", "RLIMIT_CORE", "RLIMIT_NOFILE")},
                  "env": dict(os.environ), "cwd": os.getcwd()})
            continue
        printed[0] = 0
        frame = req["frame"]
        try:
            if req["op"] == "is_goal":
                fn = glb.get("is_goal")
                if not callable(fn):
                    send({"ok": False, "kind": "exception", "message": "source does not define is_goal"})
                    continue
                out = fn(frame)
                if isinstance(out, bool):
                    send({"ok": True, "value": out})
                else:
                    send({"ok": False, "kind": "bad_output", "message": "is_goal did not return a bool: " + repr(out)[:100]})
            elif req["op"] == "legal_moves":
                fn = glb.get("legal_moves")
                if not callable(fn):
                    send({"ok": False, "kind": "exception", "message": "source does not define legal_moves"})
                    continue
                out = fn(frame)
                if valid_moves(out):
                    send({"ok": True, "moves": [dict(m) for m in out]})
                else:
                    send({"ok": False, "kind": "bad_output", "message": "legal_moves did not return a list of move dicts: " + repr(out)[:100]})
            else:
                out = predict(frame, req["move"])
                if not valid(out, len(frame), len(frame[0]) if frame else 0):
                    send({"ok": False, "kind": "bad_output", "message": "result is not a frame of the input shape: " + repr(out)[:150]})
                else:
                    send({"ok": True, "frame": list(out)})
        except Forbidden as e:
            send({"ok": False, "kind": "forbidden", "message": str(e)})
        except TooMuchOutput:
            send({"ok": False, "kind": "bad_output", "message": "printed more than %d characters" % MAX_PRINT})
        except MemoryError:
            send({"ok": False, "kind": "crash", "message": "MemoryError (memory limit)"})
            return
        except Exception as e:
            send({"ok": False, "kind": "exception", "message": (type(e).__name__ + ": " + str(e))[:300]})


main()
"""

_EMPTY: Grid = ()
_LIVE: weakref.WeakSet[SandboxedModel] = weakref.WeakSet()


@atexit.register
def _close_all() -> None:
    for model in list(_LIVE):
        model.close()


def _cut(text: str) -> str:
    return text if len(text) <= MAX_MESSAGE else text[:MAX_MESSAGE]


_encode = frame_to_rows
_decode = rows_to_frame


def _defined_functions(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    return frozenset(n.name for n in tree.body if isinstance(n, ast.FunctionDef))


def _move_dict(move: Move) -> dict[str, Any]:
    out: dict[str, Any] = {"name": move.name}
    if move.x is not None:
        out["x"] = move.x
    if move.y is not None:
        out["y"] = move.y
    return out


class _Fail(Exception):
    def __init__(self, error: SandboxError, *, kill: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.kill = kill


class SandboxedModel:
    """A `WorldModel` whose `predict` runs in an isolated worker process. Use as a context manager or
    call `close()`; workers are also reaped at garbage collection and interpreter exit."""

    def __init__(
        self,
        source: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        memory_mb: int = 512,
        cpu_seconds: int = 60,
        max_reply_bytes: int = MAX_REPLY_BYTES,
    ) -> None:
        self.source = source
        self.timeout = min(max(timeout, 0.05), HARD_TIMEOUT)
        self.last_error: SandboxError | None = None
        defined = _defined_functions(source)
        self.has_is_goal = "is_goal" in defined
        self.has_legal_moves = "legal_moves" in defined
        self.restarts = 0
        self._limits = {
            "memory": memory_mb << 20,
            "cpu": cpu_seconds,
            "nproc": 1,
            "nofile": 32,
        }
        self._max_reply = max_reply_bytes
        self._proc: subprocess.Popen[bytes] | None = None
        self._buf = bytearray()
        self._workdir = tempfile.mkdtemp(prefix="sandbox-")
        self._closed = False
        self._broken: SandboxError | None = (
            None  # the source itself is unusable: do not restart
        )
        _LIVE.add(self)
        self._ensure()

    # public ---------------------------------------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return (
            self._proc.pid
            if self._proc is not None and self._proc.poll() is None
            else None
        )

    def call(self, frame: Grid, move: Move) -> Grid | SandboxError:
        """Typed result of one prediction; never raises."""
        if self._broken is not None:
            return self._broken
        try:
            reply = self._request(
                {"op": "predict", "frame": _encode(frame), "move": _move_dict(move)}
            )
            return self._decode_reply(reply, frame)
        except _Fail as fail:
            if fail.kill:
                self._kill()
            return fail.error
        except Exception as exc:  # noqa: BLE001  the agent must never see an exception from here
            self._kill()
            return SandboxError("crash", _cut(f"{type(exc).__name__}: {exc}"))

    def predict(self, frame: Grid, move: Move) -> Grid:
        result = self.call(frame, move)
        if isinstance(result, SandboxError):
            self.last_error = result
            return _EMPTY
        return result

    def is_goal(self, frame: Grid) -> bool:
        """The source's optional `is_goal(frame) -> bool`. False when it is not defined or fails (the
        failure is recorded in `last_error`)."""
        reply = self._optional_op("is_goal", self.has_is_goal, frame)
        if isinstance(reply, SandboxError):
            self.last_error = reply
            return False
        return reply["value"] is True

    def legal_moves(self, frame: Grid) -> list[Move]:
        """The source's optional `legal_moves(frame) -> list[dict]`. Empty when it is not defined or
        fails (the failure is recorded in `last_error`)."""
        reply = self._optional_op("legal_moves", self.has_legal_moves, frame)
        if isinstance(reply, SandboxError):
            self.last_error = reply
            return []
        return [Move(m["name"], m.get("x"), m.get("y")) for m in reply["moves"]]

    def _optional_op(
        self, op: str, defined: bool, frame: Grid
    ) -> dict[str, Any] | SandboxError:
        if self._broken is not None:
            return self._broken
        if not defined:
            return SandboxError("exception", f"source does not define {op}")
        try:
            reply = self._request({"op": op, "frame": _encode(frame)})
            if reply.get("ok") is True:
                good = (
                    isinstance(reply.get("value"), bool)
                    if op == "is_goal"
                    else isinstance(reply.get("moves"), list)
                )
                if good:
                    return reply
                raise _Fail(SandboxError("bad_output", "malformed reply"), kill=True)
            self._raise_error(reply)
        except _Fail as fail:
            if fail.kill:
                self._kill()
            return fail.error
        except Exception as exc:  # noqa: BLE001  never raise into the caller
            self._kill()
            return SandboxError("crash", _cut(f"{type(exc).__name__}: {exc}"))
        raise AssertionError("unreachable")  # pragma: no cover

    def limits(self) -> dict[str, Any]:
        """Debug call: the worker's own rlimits, environment and working directory."""
        try:
            return dict(self._request({"op": "limits"}))
        except _Fail as fail:
            if fail.kill:
                self._kill()
            return {"error": str(fail.error)}

    def close(self) -> None:
        self._closed = True
        self._kill()
        shutil.rmtree(self._workdir, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass

    # internals ------------------------------------------------------------------------------------

    def _decode_reply(self, reply: dict[str, Any], frame: Grid) -> Grid | SandboxError:
        if reply.get("ok") is True:
            rows = reply.get("frame")
            if (
                isinstance(rows, list)
                and len(rows) == len(frame)
                and all(isinstance(r, str) and len(r) == len(frame[0]) for r in rows)
            ):
                try:
                    return _decode(rows)
                except ValueError:
                    pass
            raise _Fail(SandboxError("bad_output", "malformed reply"), kill=True)
        self._raise_error(reply)

    def _raise_error(self, reply: dict[str, Any]) -> NoReturn:
        kind = reply.get("kind")
        message = _cut(str(reply.get("message", "")))
        if kind not in ("exception", "bad_output", "forbidden", "crash"):
            raise _Fail(SandboxError("crash", "unknown reply from worker"), kill=True)
        # a MemoryError leaves the worker's heap suspect and it exits itself: replace it
        raise _Fail(SandboxError(kind, message), kill=kind == "crash")

    def _ensure(self) -> None:
        if self._closed or self._proc is not None and self._proc.poll() is None:
            return
        self._kill()
        env: dict[str, str] = {}
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-I", "-c", _WORKER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=self._workdir,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            self._broken = SandboxError("crash", _cut(f"cannot start worker: {exc}"))
            return
        self.restarts += 1
        init = {
            "source": self.source,
            "limits": self._limits,
            "allowed": list(ALLOWED_MODULES),
        }
        try:
            reply = self._exchange(init, max(self.timeout, 5.0))
        except _Fail as fail:
            self._kill()
            self._broken = fail.error
            self.last_error = fail.error
            return
        if reply.get("ok") is not True:
            kind = reply.get("kind")
            self._broken = SandboxError(
                kind if kind in ("exception", "forbidden", "crash") else "crash",
                _cut(str(reply.get("message", ""))),
            )
            self.last_error = self._broken
            self._kill()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure()
        if self._broken is not None:
            raise _Fail(self._broken, kill=False)
        if self._proc is None:
            raise _Fail(SandboxError("crash", "no worker"), kill=False)
        return self._exchange(payload, self.timeout)

    def _exchange(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        deadline = time.monotonic() + timeout
        try:
            proc.stdin.write(json.dumps(payload).encode() + b"\n")
            proc.stdin.flush()
        except OSError as exc:
            raise _Fail(
                SandboxError("crash", self._death(f"worker pipe closed ({exc})")),
                kill=True,
            ) from exc
        line = self._read_line(proc, deadline)
        try:
            reply = json.loads(line)
        except ValueError as exc:
            raise _Fail(
                SandboxError("bad_output", "reply is not JSON"), kill=True
            ) from exc
        if not isinstance(reply, dict):
            raise _Fail(SandboxError("bad_output", "reply is not an object"), kill=True)
        return reply

    def _read_line(self, proc: subprocess.Popen[bytes], deadline: float) -> bytes:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        buf = self._buf
        while True:
            i = buf.find(b"\n")
            if i >= 0 or len(buf) > self._max_reply:
                if i < 0 or i > self._max_reply:
                    raise _Fail(
                        SandboxError("bad_output", "reply larger than the output cap"),
                        kill=True,
                    )
                line = bytes(buf[:i])
                del buf[: i + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Fail(
                    SandboxError("timeout", f"no reply within {self.timeout:g}s"),
                    kill=True,
                )
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise _Fail(
                    SandboxError("crash", self._death("worker exited")), kill=True
                )
            buf += chunk

    def _death(self, prefix: str) -> str:
        proc = self._proc
        code: int | None = None
        if proc is not None:
            try:
                code = proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = None
        if code is not None and code < 0:
            try:
                return f"{prefix}: killed by {signal.Signals(-code).name}"
            except ValueError:
                return f"{prefix}: killed by signal {-code}"
        return f"{prefix}: exit code {code}"

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        self._buf.clear()
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.kill()
        except OSError:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def load(source: str, *, timeout: float = DEFAULT_TIMEOUT) -> SandboxedModel:
    """Check that `source` compiles and defines a top-level `predict`, then start its worker.

    Raises `SourceRejected` (with a typed `.error`) for a syntax error or a missing `predict`. Forbidden
    imports and builtins are found by the worker and show up as `last_error` of kind `forbidden` (and
    `predict` then always returns an empty grid)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SourceRejected(
            SandboxError(
                "exception", _cut(f"SyntaxError: {exc.msg} (line {exc.lineno})")
            )
        ) from exc
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "predict" for n in tree.body
    ):
        raise SourceRejected(
            SandboxError("exception", "source does not define predict(frame, move)")
        )
    return SandboxedModel(source, timeout=timeout)
