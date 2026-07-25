from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Mapping, Optional, Sequence

from jormungandr.interfaces import CommandTransport, EpisodeClient, JsonDict
from jormungandr.transport import StdioJsonTransport


class CtypesEpisode(EpisodeClient):
    """Shared-library episode client using a configurable C ABI prefix."""

    def __init__(self, lib_path: str, config: Mapping[str, Any], api_prefix: str = "episode") -> None:
        self._lib = ctypes.cdll.LoadLibrary(os.path.expandvars(lib_path))
        self._prefix = api_prefix
        self._bind_api()
        self._handle: Optional[int] = None
        self._create(config)

    def _sym(self, name: str):
        return getattr(self._lib, f"{self._prefix}_{name}")

    def _bind_api(self) -> None:
        create = self._sym("create")
        create.restype = ctypes.c_void_p
        create.argtypes = [ctypes.c_char_p]

        destroy = self._sym("destroy")
        destroy.restype = None
        destroy.argtypes = [ctypes.c_void_p]

        initialize = self._sym("initialize")
        initialize.restype = ctypes.c_int
        initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        step = self._sym("step")
        step.restype = ctypes.c_int
        step.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        last_step_json = self._sym("last_step_json")
        last_step_json.restype = ctypes.c_char_p
        last_step_json.argtypes = [ctypes.c_void_p]

        last_error = self._sym("last_error")
        last_error.restype = ctypes.c_char_p
        last_error.argtypes = [ctypes.c_void_p]

        is_terminal = self._sym("is_terminal")
        is_terminal.restype = ctypes.c_int
        is_terminal.argtypes = [ctypes.c_void_p]

        remaining_steps = self._sym("remaining_steps")
        remaining_steps.restype = ctypes.c_longlong
        remaining_steps.argtypes = [ctypes.c_void_p]

    def _create(self, config: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(config)).encode("utf-8")
        handle = self._sym("create")(payload)
        if not handle:
            raise RuntimeError(self.last_error() or "create failed")
        self._handle = handle

    def initialize(self, init_params: Optional[Mapping[str, Any]] = None) -> None:
        payload = json.dumps(dict(init_params or {})).encode("utf-8")
        rc = self._sym("initialize")(self._handle, payload)
        if rc != 0:
            raise RuntimeError(self.last_error() or f"initialize failed: {rc}")

    def step(self, step_params: Optional[Mapping[str, Any]] = None) -> JsonDict:
        payload = json.dumps(dict(step_params or {})).encode("utf-8")
        rc = self._sym("step")(self._handle, payload)
        if rc != 0:
            raise RuntimeError(self.last_error() or f"step failed: {rc}")
        return self.last_step()

    def last_step(self) -> JsonDict:
        raw = self._sym("last_step_json")(self._handle)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def last_error(self) -> str:
        raw = self._sym("last_error")(self._handle)
        if not raw:
            return ""
        return raw.decode("utf-8")

    def is_terminal(self) -> bool:
        return bool(self._sym("is_terminal")(self._handle))

    def remaining_steps(self) -> Optional[int]:
        return int(self._sym("remaining_steps")(self._handle))

    def close(self) -> None:
        if self._handle:
            self._sym("destroy")(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CtypesEpisodeFactory:
    def __init__(self, lib_path: str, api_prefix: str = "episode") -> None:
        self._lib_path = lib_path
        self._api_prefix = api_prefix

    def create(self, config: Mapping[str, Any]) -> EpisodeClient:
        return CtypesEpisode(self._lib_path, config, api_prefix=self._api_prefix)


class CommandEpisode(EpisodeClient):
    """Episode client over a command transport using JSON RPC-style ops."""

    def __init__(self, transport: CommandTransport, config: Mapping[str, Any]) -> None:
        self._transport = transport
        self._transport.call({"op": "create", "config": dict(config)})

    def initialize(self, init_params: Optional[Mapping[str, Any]] = None) -> None:
        self._transport.call({"op": "initialize", "params": dict(init_params or {})})

    def step(self, step_params: Optional[Mapping[str, Any]] = None) -> JsonDict:
        out = self._transport.call({"op": "step", "params": dict(step_params or {})})
        return out if isinstance(out, dict) else {"result": out}

    def is_terminal(self) -> bool:
        out = self._transport.call({"op": "is_terminal"})
        if isinstance(out, dict):
            if "terminal" in out:
                return bool(out["terminal"])
            if "result" in out:
                return bool(out["result"])
        return bool(out)

    def remaining_steps(self) -> Optional[int]:
        out = self._transport.call({"op": "remaining_steps"})
        if isinstance(out, dict):
            if "remaining_steps" in out:
                return int(out["remaining_steps"])
            if "result" in out:
                return int(out["result"])
        if out is None:
            return None
        return int(out)

    def close(self) -> None:
        try:
            self._transport.call({"op": "close"})
        except Exception:
            pass
        self._transport.close()


class SubprocessEpisodeFactory:
    """Spawns a new process per episode and talks over JSON lines on stdio."""

    def __init__(
        self,
        command: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._command = list(command)
        self._cwd = cwd
        self._env = dict(env) if env is not None else None

    def create(self, config: Mapping[str, Any]) -> EpisodeClient:
        transport = StdioJsonTransport(self._command, cwd=self._cwd, env=self._env)
        return CommandEpisode(transport, config)
