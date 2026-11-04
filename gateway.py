#!/usr/bin/env python3
"""Shared, OpenAI-compatible LLM gateway for a private network.

One endpoint that any client can point an OpenAI SDK at:

    http://<host>:9302/v1

Upstream is OpenRouter. Provider routing is NOT hardcoded here: it is read live
from a routing file (`{"model": {"only": [...], "order": [...]}}` JSON, or a
Hermes-style config.yaml with `provider_routing.models.<model>`), re-read
whenever that file changes. A price-routing job that rewrites the file therefore
updates every client of this gateway on their next request — no restart.
Models without a routing entry are passed through unchanged.

Pure stdlib. Endpoints: /health, /v1/models, /v1/models/<id>,
/v1/chat/completions, /v1/completions, /v1/embeddings, and any other POST
/v1/* is passed through.

Env:
  OPENROUTER_API_KEY   upstream key (falls back to ~/.hermes/.env)
  DEFAULT_MODEL        model used when a request omits "model"
  PROXY_API_KEY        if set, clients must send it as a Bearer token
  GATEWAY_HOSTS        comma list of bind addresses (default: tailscale IP,127.0.0.1)
  GATEWAY_PORT         default 9302
  GATEWAY_ROUTING_FILE yaml/json routing table (default: ~/.hermes/config.yaml if present)
  HERMES_CONFIG        legacy alias for GATEWAY_ROUTING_FILE
  GATEWAY_REFERER      OpenRouter HTTP-Referer attribution header
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-v4.1-flash-1017")
PORT = int(os.environ.get("GATEWAY_PORT", "9302"))
REFERER = os.environ.get("GATEWAY_REFERER", "https://github.com/vikarag/llm-gateway")
X_TITLE = os.environ.get("GATEWAY_X_TITLE", "llm-gateway")
# Routing source: an explicit file wins; otherwise use Hermes' config if it
# exists (the origin of this tool); otherwise no routing (pure passthrough).
CONFIG_PATH = os.path.expanduser(
    os.environ.get("GATEWAY_ROUTING_FILE")
    or os.environ.get("HERMES_CONFIG")
    or ("~/.hermes/config.yaml" if os.path.exists(os.path.expanduser("~/.hermes/config.yaml")) else ""))
STATE_DIR = os.path.expanduser("~/.local/state/llm-gateway")
LOG_PATH = os.path.join(STATE_DIR, "requests.jsonl")
UPSTREAM_HOST = "openrouter.ai"
UPSTREAM_BASE = "/api/v1"
READ_TIMEOUT = 600
START = time.time()

_log_lock = threading.Lock()
_stats_lock = threading.Lock()
STATS = {"requests": 0, "errors": 0, "streams": 0, "by_model": {}, "last_provider": None}


# --------------------------------------------------------------------------- env
ENV_CANDIDATES = (
    "~/.config/llm-gateway/env",   # 1. preferred: this gateway's own key
    "~/llm-gateway/.env",          # 2. convenience copy next to the code
    "~/.hermes/.env",              # 3. last resort: shared Hermes key
)
KEY_SOURCE = "unset"


def _read_key_path(path: str) -> str | None:
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    value = v.strip().strip('"').strip("'")
                    if value:
                        return value
    except OSError:
        pass
    return None


def _has_commented_key(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("#") and "OPENROUTER_API_KEY" in s and "=" in s:
                    return True
    except OSError:
        pass
    return False


def _load_env() -> None:
    """Resolve the upstream key, most specific source first.

    Precedence: systemd Environment= > ~/.config/llm-gateway/env >
    ~/llm-gateway/.env > ~/.hermes/.env (the key Hermes itself uses).
    """
    global KEY_SOURCE
    if os.environ.get("OPENROUTER_API_KEY"):
        KEY_SOURCE = "environment"
        return
    for path in ENV_CANDIDATES:
        value = _read_key_path(path)
        if value:
            os.environ["OPENROUTER_API_KEY"] = value
            KEY_SOURCE = os.path.expanduser(path)
            return
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded) and _has_commented_key(expanded):
            # A key pasted onto the template's commented line is skipped, and the
            # gateway would otherwise fall back to the Hermes key in silence.
            print(f"WARN: {expanded} has an OPENROUTER_API_KEY line that is "
                  f"commented out (#) — it is being ignored", file=sys.stderr, flush=True)


def key_fingerprint() -> str:
    """Short, non-reversible id so you can tell WHICH key is loaded without
    printing it: first 8 hex of sha256 plus the key's last 4 characters."""
    import hashlib

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return "none"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{digest}…{key[-4:]}"


# ----------------------------------------------------------------- routing table
ROUTING_KEYS = ("only", "ignore", "order", "sort", "require_parameters", "data_collection")


def _clean_models(models) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not isinstance(models, dict):
        return out
    for model, cfg in models.items():
        if not isinstance(cfg, dict):
            continue
        clean = {k: cfg[k] for k in ROUTING_KEYS if cfg.get(k) is not None}
        if clean:
            out[str(model)] = clean
    return out


class RoutingTable:
    """Live view of a routing file, re-read whenever its mtime/size changes.

    Accepts either a flat JSON map:

        {"<model>": {"only": [...], "order": [...]}}
        {"models": {"<model>": {...}}}

    or a Hermes-style config.yaml with `provider_routing.models.<model>`.
    """

    def __init__(self, path: str):
        self.path = path
        self._stamp: tuple | None = None
        self._map: dict[str, dict] = {}

    def _parse(self) -> dict[str, dict]:
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return {}
        if self.path.endswith(".json") or text.lstrip().startswith("{"):
            try:
                data = json.loads(text)
                return _clean_models((data or {}).get("models") or data)
            except Exception:
                return {}
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
            out = _clean_models(((data.get("provider_routing") or {}).get("models") or {}))
            if out:
                return out
        except Exception:
            pass
        return self._parse_fallback(text)

    @staticmethod
    def _parse_fallback(text: str) -> dict[str, dict]:
        """Minimal parser used only if PyYAML is unavailable/broken.

        Indentation-relative rather than fixed-width: `models:` may sit at any
        depth, and its children are derived from it. (A fixed 4/6/8/10 layout
        silently returns nothing for the common 2-space style.)
        """
        out: dict[str, dict] = {}
        lines = text.splitlines()
        start = next((i for i, l in enumerate(lines) if re.match(r"^provider_routing:\s*$", l)), None)
        if start is None:
            return out
        models_indent = None
        cur_model = None
        cur_key = None
        for line in lines[start + 1:]:
            if not line.strip():
                continue
            if not line[0].isspace():
                break  # end of the provider_routing block
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped == "models:":
                models_indent = indent
                continue
            if models_indent is None:
                continue
            if indent == models_indent + 2 and not stripped.startswith("- "):
                cur_model = stripped.rstrip(":").strip().strip('"').strip("'")
                out.setdefault(cur_model, {})
                cur_key = None
                continue
            if cur_model and indent == models_indent + 4 and not stripped.startswith("- "):
                if stripped.endswith(":"):
                    cur_key = stripped.rstrip(":").strip()
                    out[cur_model][cur_key] = [] if cur_key in ("only", "ignore", "order") else None
                elif ":" in stripped:
                    k, v = stripped.split(":", 1)
                    out[cur_model][k.strip()] = v.strip().strip('"').strip("'") in ("true", "True", "yes")
                continue
            if cur_model and cur_key and indent >= models_indent + 6 and stripped.startswith("- "):
                val = stripped[2:].strip().strip('"').strip("'")
                if isinstance(out[cur_model].get(cur_key), list):
                    out[cur_model][cur_key].append(val)
        return {m: c for m, c in out.items() if any(k in c for k in ("only", "order"))}

    def get(self, model: str) -> dict | None:
        if not self.path:
            return None
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        # mtime alone is too coarse for two writes in the same second.
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp != self._stamp:
            self._map = self._parse()
            self._stamp = stamp
        cfg = self._map.get(model)
        return dict(cfg) if cfg else None

    def models(self) -> list[str]:
        self.get(DEFAULT_MODEL)  # force refresh if stale
        return sorted(self._map)

    def resolve(self, name: str) -> str:
        """Map a client-supplied model id onto a routing-table key.

        Clients routinely send the bare slug (`deepseek-v4.1-flash-1017`)
        while the table is keyed by the fully-qualified id
        (`deepseek/deepseek-v4.1-flash-1017`). A miss here silently drops the
        `provider` block and lets OpenRouter pick any upstream, so match on the
        last path segment too. Ambiguous matches are left untouched.
        """
        self.get(DEFAULT_MODEL)  # force refresh if stale
        if not name or name in self._map:
            return name
        want = name.strip().lower()
        hits = [k for k in self._map if k.rsplit("/", 1)[-1].lower() == want]
        return hits[0] if len(hits) == 1 else name


ROUTING = RoutingTable(CONFIG_PATH)


# ------------------------------------------------------------------------ helpers
def log_record(rec: dict) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


def bump(model: str, ok: bool, stream: bool, provider: str | None = None) -> None:
    with _stats_lock:
        STATS["requests"] += 1
        if not ok:
            STATS["errors"] += 1
        if stream:
            STATS["streams"] += 1
        STATS["by_model"][model] = STATS["by_model"].get(model, 0) + 1
        if provider:
            STATS["last_provider"] = provider


TAILSCALE_BIN = os.environ.get("TAILSCALE_BIN", "tailscale")
SUPERVISE_SEC = int(os.environ.get("GATEWAY_SUPERVISE_SEC", "15"))


def tailscale_ip() -> str | None:
    try:
        out = subprocess.run([TAILSCALE_BIN, "ip", "-4"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.split():
            if line.count(".") == 3:
                return line
    except Exception:
        pass
    return None


def desired_hosts() -> list[str]:
    """127.0.0.1 always; the tailscale address whenever it exists.

    Re-evaluated continuously, not once at startup: on a reboot tailscaled can
    take longer than any fixed wait, and a one-shot bind would then leave the
    gateway reachable only on localhost until someone restarted it.
    """
    env = os.environ.get("GATEWAY_HOSTS")
    if env:
        return [h.strip() for h in env.split(",") if h.strip()]
    hosts = ["127.0.0.1"]
    ip = tailscale_ip()
    if ip and ip not in hosts:
        hosts.insert(0, ip)
    return hosts


class Binder:
    """Keeps a listening socket on every address that should have one."""

    def __init__(self) -> None:
        self.servers: dict[str, ThreadingHTTPServer] = {}
        self._lock = threading.Lock()

    def _start(self, host: str) -> bool:
        try:
            httpd = ThreadingHTTPServer((host, PORT), Handler)
        except OSError as exc:
            print(f"WARN: cannot bind {host}:{PORT} — {exc}", file=sys.stderr, flush=True)
            return False
        httpd.daemon_threads = True
        self.servers[host] = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True, name=f"http-{host}").start()
        print(f"llm-gateway listening on http://{host}:{PORT}/v1", flush=True)
        return True

    def sync(self) -> None:
        desired = desired_hosts()
        with self._lock:
            for host in desired:
                if host not in self.servers:
                    self._start(host)
            for host in list(self.servers):
                if host not in desired and host != "127.0.0.1":
                    print(f"llm-gateway dropping {host}:{PORT} (address gone)", flush=True)
                    try:
                        self.servers.pop(host).shutdown()
                    except Exception:
                        pass

    def hosts(self) -> list[str]:
        with self._lock:
            return sorted(self.servers)

    def run_forever(self) -> None:
        # At boot, retry hard for a minute before settling into the slow poll.
        for _ in range(30):
            self.sync()
            if os.environ.get("GATEWAY_HOSTS") or len(self.servers) > 1:
                break
            time.sleep(2)
        while True:
            time.sleep(SUPERVISE_SEC)
            try:
                self.sync()
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: binder sync failed: {exc}", file=sys.stderr, flush=True)


BINDER = Binder()


def resolve_model(requested: str | None) -> str:
    if not requested or requested in ("default", "auto"):
        return DEFAULT_MODEL
    # Normalise bare slugs onto the routing table's fully-qualified ids so the
    # provider whitelist is never silently skipped.
    return ROUTING.resolve(str(requested).strip())


# ------------------------------------------------------------------------ handler
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llm-gateway/1.0"

    # ---- plumbing
    def _json(self, payload, code: int = 200, extra_headers: dict | None = None) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, message: str, code: int = 400, etype: str = "invalid_request_error") -> None:
        self._json({"error": {"message": message, "type": etype}}, code)

    def _authorized(self) -> bool:
        required = os.environ.get("PROXY_API_KEY")
        if not required:
            return True
        supplied = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        if not supplied:
            supplied = (self.headers.get("x-api-key") or "").strip()
        return supplied == required

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _client(self) -> str:
        return self.client_address[0] if self.client_address else "?"

    def log_message(self, fmt, *args):  # keep the journal quiet
        return

    # ---- GET
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            with _stats_lock:
                stats = dict(STATS)
            routing = ROUTING.get(DEFAULT_MODEL) or {}
            self._json({
                "status": "ok",
                "uptime_s": int(time.time() - START),
                "default_model": DEFAULT_MODEL,
                "routing_order": routing.get("order", []),
                "routing_only_count": len(routing.get("only", []) or []),
                "auth_required": bool(os.environ.get("PROXY_API_KEY")),
                "upstream_key_from": KEY_SOURCE,
                "upstream_key_id": key_fingerprint(),
                "routing_file": CONFIG_PATH or None,
                "listening_on": BINDER.hosts(),
                "stats": stats,
            })
            return
        if path in ("/v1/models", "/models"):
            ids = [DEFAULT_MODEL] + [m for m in ROUTING.models() if m != DEFAULT_MODEL]
            seen, data = set(), []
            for mid in ids:
                if mid in seen:
                    continue
                seen.add(mid)
                rt = ROUTING.get(mid) or {}
                data.append({
                    "id": mid,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "openrouter",
                    "provider_count": len(rt.get("only", []) or []) or None,
                    "preferred_providers": rt.get("order", []),
                })
            self._json({"object": "list", "data": data})
            return
        if path.startswith("/v1/models/"):
            mid = path[len("/v1/models/"):]
            rt = ROUTING.get(mid)
            self._json({"id": mid, "object": "model", "created": 1700000000,
                        "owned_by": "openrouter", "provider_routing": rt})
            return
        if path == "/":
            self._json({
                "name": "llm-gateway",
                "openai_base_url": f"http://{self.headers.get('Host') or 'localhost:' + str(PORT)}/v1",
                "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
                "default_model": DEFAULT_MODEL,
            })
            return
        self._error(f"not found: {path}", 404)

    # ---- POST
    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorized():
            self._error("invalid or missing API key", 401, "authentication_error")
            return
        if not path.startswith("/v1/"):
            self._error(f"not found: {path}", 404)
            return

        body = self._read_body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._error("request body is not valid JSON")
            return
        if not isinstance(data, dict):
            self._error("request body must be a JSON object")
            return

        stream = bool(data.get("stream"))
        model = resolve_model(data.get("model"))

        # Apply the live fp8 routing table for this model, if one exists.
        routing = ROUTING.get(model)
        if routing:
            data["provider"] = routing
        data["model"] = model
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self._upstream_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": REFERER,
            "X-Title": X_TITLE,
        }

        started = time.time()
        try:
            conn = http.client.HTTPSConnection(UPSTREAM_HOST, timeout=READ_TIMEOUT)
            conn.request("POST", UPSTREAM_BASE + path[3:] if path.startswith("/v1/") else path,
                         body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001
            bump(model, False, stream)
            log_record({"ts": time.time(), "client": self._client(), "model": model,
                        "status": 599, "error": str(exc)})
            self._error(f"upstream connection failed: {exc}", 502, "api_error")
            return

        if resp.status >= 400:
            raw = resp.read()
            bump(model, False, stream)
            log_record({"ts": time.time(), "client": self._client(), "model": model,
                        "status": resp.status, "error": raw[:400].decode("utf-8", "replace")})
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"error": {"message": raw.decode("utf-8", "replace"), "type": "api_error"}}
            self._json(payload, resp.status)
            return

        if stream:
            self._relay_stream(resp, model, started)
        else:
            self._relay_full(resp, model, started)

    # ---- upstream helpers
    def _upstream_key(self) -> str:
        """Host key by default. A client that brings its own OpenRouter key
        (Authorization: Bearer sk-or-...) is billed on that key instead — handy
        for tailnet devices that want their own quota/credits."""
        supplied = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        if supplied.startswith("sk-or-") and not os.environ.get("PROXY_API_KEY"):
            return supplied
        return os.environ.get("OPENROUTER_API_KEY", "")

    def _relay_full(self, resp, model: str, started: float) -> None:
        raw = resp.read()
        provider, usage = None, None
        try:
            payload = json.loads(raw)
            provider = payload.get("provider")
            usage = payload.get("usage")
        except Exception:
            payload = None
        ms = int((time.time() - started) * 1000)
        bump(model, True, False, provider)
        log_record({"ts": time.time(), "client": self._client(), "model": model, "stream": False,
                    "status": resp.status, "ms": ms, "provider": provider,
                    "usage": usage})
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type") or "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if provider:
            self.send_header("X-Gateway-Provider", str(provider))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _relay_stream(self, resp, model: str, started: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        provider, usage, ok, sent = None, None, True, 0

        def chunk(payload: bytes) -> None:
            if not payload:
                return
            self.wfile.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
            self.wfile.flush()

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(b"data:"):
                    payload = stripped[5:].strip()
                    if payload == b"[DONE]":
                        chunk(b"data: [DONE]\n\n")
                        break
                    try:
                        obj = json.loads(payload)
                        provider = obj.get("provider") or provider
                        if obj.get("usage"):
                            usage = obj["usage"]
                    except Exception:
                        pass
                    chunk(b"data: " + payload + b"\n\n")
                    sent += 1
                elif stripped.startswith(b":"):
                    chunk(stripped + b"\n\n")  # keepalive comment
        except (BrokenPipeError, ConnectionResetError):
            ok = False
        except Exception as exc:  # noqa: BLE001
            ok = False
            try:
                err = json.dumps({"error": {"message": f"stream aborted: {exc}"}}).encode()
                chunk(b"data: " + err + b"\n\n")
            except Exception:
                pass
        finally:
            try:
                chunk(b"")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass
            try:
                resp.close()
            except Exception:
                pass
        # A client that hangs up mid-stream is normal (Ctrl-C, read timeout on the
        # far end) — log it as 499 but do not count it as a gateway error.
        bump(model, ok or sent > 0, True, provider)
        log_record({"ts": time.time(), "client": self._client(), "model": model, "stream": True,
                    "status": 200 if ok else 499, "ms": int((time.time() - started) * 1000),
                    "provider": provider, "chunks": sent, "usage": usage})


# --------------------------------------------------------------------------- main
def main() -> int:
    _load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("FATAL: no OPENROUTER_API_KEY (env or ~/.hermes/.env)", file=sys.stderr)
        return 1
    os.makedirs(STATE_DIR, exist_ok=True)

    BINDER.sync()
    if not BINDER.hosts():
        print("FATAL: could not bind any address", file=sys.stderr)
        return 1

    routing = ROUTING.get(DEFAULT_MODEL) or {}
    print(f"default model: {DEFAULT_MODEL}", flush=True)
    print(f"routing order: {routing.get('order', [])}", flush=True)

    threading.Thread(target=BINDER.run_forever, daemon=True, name="binder").start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        for httpd in list(BINDER.servers.values()):
            httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
