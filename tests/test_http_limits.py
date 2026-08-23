from __future__ import annotations

import gzip
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jormungandr.service import JormungandrHttpServer, JormungandrRuntime


def _post_raw(url: str, body: bytes, *, encoding: str = "") -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if encoding:
        headers["Content-Encoding"] = encoding
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=5.0) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_http_rejects_compressed_and_expanded_bodies_over_limits() -> None:
    runtime = JormungandrRuntime(inference_replicas=1)
    server = JormungandrHttpServer(
        "127.0.0.1",
        0,
        runtime=runtime,
        max_compressed_request_bytes=64,
        max_decompressed_request_bytes=128,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/models"
    try:
        status, payload = _post_raw(url, b"{" + b" " * 80 + b"}")
        assert status == 413
        assert payload["ok"] is False
        expanded = json.dumps({"metadata": {"padding": "x" * 512}}).encode()
        compressed = gzip.compress(expanded)
        assert len(compressed) < 64
        status, payload = _post_raw(url, compressed, encoding="gzip")
        assert status == 413
        assert "decompressed" in payload["error"]
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()


def test_http_accepts_bounded_gzip_json() -> None:
    runtime = JormungandrRuntime(inference_replicas=1)
    server = JormungandrHttpServer(
        "127.0.0.1",
        0,
        runtime=runtime,
        max_compressed_request_bytes=1024,
        max_decompressed_request_bytes=1024,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/models"
    body = gzip.compress(json.dumps({"obs_dim": 2}).encode())
    try:
        status, payload = _post_raw(url, body, encoding="gzip")
        assert status == 201
        assert payload["model"]["obs_dim"] == 2
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()
