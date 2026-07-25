from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence

from jormungandr.interfaces import CommandTransport, JsonDict


class StdioJsonTransport(CommandTransport):
    """JSON request/response transport over stdio (one JSON object per line)."""

    def __init__(
        self,
        command: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._command = list(command)
        self._proc = subprocess.Popen(
            self._command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def call(self, request: Mapping[str, Any]) -> JsonDict:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("stdio pipes are not available")
        if self._proc.poll() is not None:
            raise RuntimeError(f"transport process exited with code {self._proc.returncode}")

        payload = json.dumps(dict(request), separators=(",", ":"))
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("transport returned EOF")

        try:
            data: JsonDict = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON response: {line!r}") from exc

        if isinstance(data, dict) and data.get("ok") is False:
            raise RuntimeError(str(data.get("error") or "transport request failed"))
        if isinstance(data, dict) and "result" in data:
            result = data.get("result")
            if isinstance(result, dict):
                return result
            return {"result": result}
        return data

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2.0)
        except Exception:
            self._proc.kill()


class AeronPubSubTransport(CommandTransport):
    """Extension point for Aeron-backed request/response transport."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Aeron transport is not implemented yet. "
            "Provide a CommandTransport with call()/close() and reuse CommandEpisode."
        )

    def call(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
