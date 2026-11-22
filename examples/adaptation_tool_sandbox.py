"""Loopback HTTP tools with typed arguments and independently checked outputs.

Fault injection belongs to this example server, never to the policy input.
No arbitrary code execution, external credentials, or external API is needed.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SortCall:
    service: str
    values: tuple[int, ...]

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) != {"service", "values"}:
            raise ValueError("expected service and values")
        if not isinstance(raw["service"], str) or not raw["service"]:
            raise ValueError("service must be a nonempty string")
        values = raw["values"]
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            raise ValueError("values must contain 1..64 integers")
        if any(type(v) is not int or abs(v) > 1000000 for v in values):
            raise ValueError("values must be bounded integers")
        return cls(raw["service"], tuple(values))


def verify(call, response):
    # Check ordering AND multiset, independently of the service implementation.
    values = response.get("values") if isinstance(response, dict) else None
    return bool(
        isinstance(values, list)
        and all(type(x) is int for x in values)
        and len(values) == len(call.values)
        and all(a <= b for a, b in zip(values, values[1:]))
        and all(values.count(x) == call.values.count(x) for x in set(call.values))
    )


@contextmanager
def tool_server(service_ids, outcome):
    """outcome(service_id) is private simulator state, not an HTTP argument."""
    allowed = frozenset(service_ids)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/tools/sort" or not 0 < length <= 8192:
                    raise ValueError("invalid path or request size")
                call = SortCall.parse(json.loads(self.rfile.read(length)))
                if call.service not in allowed:
                    raise ValueError("unknown service")
                good = outcome(call.service)
                # Real computation, with controlled incorrect-output faults.
                result = list(call.values)
                for i in range(1, len(result)):
                    j = i
                    while j and result[j - 1] > result[j]:
                        result[j - 1], result[j] = result[j], result[j - 1]
                        j -= 1
                if not good:
                    result[0] -= 1  # always violates the input multiset
                payload, status = {"values": result}, 200
            except (ValueError, TypeError, json.JSONDecodeError):
                payload, status = {"error": "invalid tool arguments"}, 400
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/tools/sort"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def invoke(url, call):
    payload = json.dumps(
        {"service": call.service, "values": list(call.values)}
    ).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as result:
        response = json.loads(result.read(8192))
    return float(verify(call, response)), response
