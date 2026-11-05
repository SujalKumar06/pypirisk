"""Side HTTP server that gives the Streamlit frontend ``/health`` and ``/metrics``.

Streamlit cannot add custom routes to its own server, so these two endpoints run on a
separate port inside the same process.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)


class OpsHandler(BaseHTTPRequestHandler):
    """Answer ``GET /health`` and ``GET /metrics``; everything else is 404."""

    def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
        """Serve the health document or the Prometheus metrics."""
        if self.path == "/health":
            self._reply(200, "application/json", json.dumps({"status": "ok"}).encode())
        elif self.path == "/metrics":
            self._reply(200, CONTENT_TYPE_LATEST, generate_latest())
        else:
            self._reply(404, "text/plain", b"not found")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base class signature
        """Route access logs through ``logging`` at debug level instead of stderr."""
        logger.debug(format, *args)

    def _reply(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_ops_server(port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Start the operations server in a daemon thread.

    Args:
        port: Port to listen on.
        host: Interface to bind.

    Returns:
        The running server, so callers can shut it down.
    """
    server = ThreadingHTTPServer((host, port), OpsHandler)
    Thread(target=server.serve_forever, name="ops-server", daemon=True).start()
    logger.info("Operations server started", extra={"port": port})
    return server
