from __future__ import annotations

import json
import mimetypes
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .adapters import build_registry


class ActivationStormApp:
    def __init__(self, static_dir: Path, registry: dict | None = None) -> None:
        self.static_dir = static_dir
        self.registry = registry if registry is not None else build_registry()

    def models_payload(self) -> dict:
        models = [adapter.model_info().to_dict() for adapter in self.registry.values()]
        default_model = models[0]["id"] if models else None
        return {"models": models, "default_model": default_model}

    def analyze(self, payload: dict) -> dict:
        model_id = payload.get("model_id")
        prompt = payload.get("prompt", "")
        if not model_id:
            raise ValueError("model_id is required.")
        if model_id not in self.registry:
            raise ValueError(f"Unknown model_id: {model_id}")
        include_special_tokens = bool(payload.get("include_special_tokens", False))
        return self.registry[model_id].analyze_prompt(
            prompt,
            include_special_tokens=include_special_tokens,
            include_layer_analysis=False,
        ).to_dict()

    def layer_analysis_payload(self, payload: dict) -> dict:
        model_id = payload.get("model_id")
        prompt = payload.get("prompt", "")
        if not model_id:
            raise ValueError("model_id is required.")
        if model_id not in self.registry:
            raise ValueError(f"Unknown model_id: {model_id}")
        include_special_tokens = bool(payload.get("include_special_tokens", False))
        result = self.registry[model_id].analyze_prompt(
            prompt,
            include_special_tokens=include_special_tokens,
            include_layer_analysis=True,
        ).to_dict()
        return {
            "target_position": result["target_position"],
            "target_token_id": result["target_token_id"],
            "target_token": result["target_token"],
            "layer_analysis": result["layer_analysis"],
        }

    def architecture_payload(self, model_id: str) -> dict:
        if not model_id:
            raise ValueError("model_id is required.")
        if model_id not in self.registry:
            raise ValueError(f"Unknown model_id: {model_id}")
        adapter = self.registry[model_id]
        return {
            "model": adapter.model_info().to_dict(),
            "architecture": adapter.architecture_text(),
        }


class ActivationStormHandler(BaseHTTPRequestHandler):
    server_version = "ActivationStorm/0.1"

    @property
    def app(self) -> ActivationStormApp:
        return self.server.app

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/models":
            self._send_json(HTTPStatus.OK, self.app.models_payload())
            return
        if path == "/api/architecture":
            try:
                model_id = parse_qs(parsed.query).get("model_id", [""])[0]
                self._send_json(HTTPStatus.OK, self.app.architecture_payload(model_id))
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/analyze", "/api/layer-analysis"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw or b"{}")
            if path == "/api/analyze":
                result = self.app.analyze(payload)
            else:
                result = self.app.layer_analysis_payload(payload)
            self._send_json(HTTPStatus.OK, result)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _serve_static(self, path: str) -> None:
        requested = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (self.app.static_dir / requested).resolve()
        if not str(target).startswith(str(self.app.static_dir.resolve())) or not target.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        mime_type, _ = mimetypes.guess_type(str(target))
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):  # noqa: A003
        return

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ActivationStormServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, app: ActivationStormApp):
        super().__init__(server_address, handler_cls)
        self.app = app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    static_dir = Path(__file__).with_name("static")
    app = ActivationStormApp(static_dir=static_dir)
    server = ActivationStormServer((host, port), ActivationStormHandler, app)
    print(f"Activation Storm running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
