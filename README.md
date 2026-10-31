# llm-gateway

A small, dependency-free OpenAI-compatible proxy that shares **one** LLM endpoint with
everything on your private network (Tailscale, WireGuard, LAN) — with per-model provider
routing that can be changed while it runs.

```
your device ──► http://gateway-host:9302/v1 ──► OpenRouter ──► cheapest healthy provider
```

Any OpenAI SDK works. The point is that provider selection lives in *one* file on one
machine instead of being duplicated into every client — and it is re-read on every
request, so a cron job can re-price your traffic twice a day without restarting anything.

```
$ curl -s http://gateway-host:9302/health
{"status":"ok","default_model":"deepseek/deepseek-v4.1-flash-1017",
 "routing_order":["morph/fp8","deepinfra/fp8","inference-net/fp8","streamlake/fp8"],
 "routing_only_count":17,"listening_on":["100.x.y.z","127.0.0.1"], ...}
```

## Why not just share the API key?

Because then every device picks its own provider and you cannot answer "which upstream is
this request hitting, and what did it cost?". Here the routing decision is server-side,
requests are logged with the provider that actually served them, and changing the policy
is one file edit — no client reconfiguration.

## Features

- **OpenAI-compatible**: `/v1/chat/completions` (streaming and not), `/v1/completions`,
  `/v1/embeddings`, `/v1/models`, plus pass-through for any other `/v1/*` POST.
- **Live routing table**: `only` / `order` / `ignore` / `sort` / `require_parameters` /
  `data_collection` per model, read from a JSON file or a Hermes-style `config.yaml`,
  re-read whenever the file changes (no restart, no reload endpoint).
- **Streaming done properly**: real HTTP/1.1 chunked framing and SSE relay, including
  upstream keep-alive comments.
- **Per-request audit log**: client, model, latency, tokens, cost, and the upstream
  provider that served the request.
- **Network-aware binding**: binds the Tailscale address *and* loopback, and keeps
  checking — a slow `tailscaled` at boot does not leave it loopback-only.
- **No dependencies**: Python 3.10+ standard library only. PyYAML is used if present,
  with a fallback parser if it is not.
- **Optional shared secret** for clients, and optional per-client OpenRouter keys.

## Quick start

```bash
git clone https://github.com/vikarag/llm-gateway.git
cd llm-gateway
cp .env.example .env          # then put your OpenRouter key in it
python3 gateway.py
```

```bash
curl -s http://127.0.0.1:9302/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://gateway-host:9302/v1", api_key="local")
print(client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash-1017",
    messages=[{"role": "user", "content": "hi"}],
).choices[0].message.content)
```

## Routing

`GATEWAY_ROUTING_FILE` points at either a JSON map:

```json
{
  "deepseek/deepseek-v4.1-flash-1017": {
    "only":  ["morph/fp8", "deepinfra/fp8", "streamlake/fp8"],
    "order": ["morph/fp8", "deepinfra/fp8", "streamlake/fp8"]
  }
}
```

…or a YAML file with the same shape under `provider_routing.models` (which is what Hermes
writes). See `examples/routing.json`. If no routing entry matches the requested model, the
request is forwarded untouched.

`only` is a hard whitelist of OpenRouter endpoint tags; `order` is a preference inside it.
Fallbacks stay enabled unless you also set `allow_fallbacks: false`.

A price-refresh job only has to rewrite that file — every client picks it up on its next
request, because the gateway keeps no cached copy of the policy.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | status, live routing, key id (hashed), listen addresses, counters |
| GET | `/v1/models` | default model + every model with a routing entry |
| GET | `/v1/models/<id>` | the routing applied to one model |
| POST | `/v1/chat/completions` | streaming + non-streaming |
| POST | `/v1/completions`, `/v1/embeddings`, `/v1/*` | passed through |

## Keys

Three ways to authenticate, in order of precedence:

1. `PROXY_API_KEY` — if set, clients must send it as `Authorization: Bearer …`; the
   gateway uses its own upstream key.
2. A client may send its own `sk-or-…` key and be billed on that key (when
   `PROXY_API_KEY` is unset).
3. Otherwise the gateway's own key is used, from `OPENROUTER_API_KEY`, or from the first
   of `~/.config/llm-gateway/env`, `./.env`, `~/.hermes/.env`.

`GET /health` reports `upstream_key_from` and a non-reversible `upstream_key_id`
(sha256 prefix + last 4 chars), so you can confirm which key is live without printing it.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | upstream key (see Keys) |
| `DEFAULT_MODEL` | `deepseek/deepseek-v4.1-flash-1017` | used when a request omits `model` |
| `GATEWAY_PORT` | `9302` | listen port |
| `GATEWAY_HOSTS` | auto: tailscale IP + `127.0.0.1` | explicit comma-separated bind list |
| `GATEWAY_ROUTING_FILE` | `~/.hermes/config.yaml` if present | routing table (JSON or YAML) |
| `PROXY_API_KEY` | unset | shared secret required from clients |
| `GATEWAY_REFERER` / `GATEWAY_X_TITLE` | repo URL / `llm-gateway` | OpenRouter attribution headers |
| `GATEWAY_SUPERVISE_SEC` | `15` | address re-check interval |
| `TAILSCALE_BIN` | `tailscale` | binary used to discover the tailnet address |

## Run it as a service

```bash
sudo cp deploy/llm-gateway@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-gateway@$USER
journalctl -u llm-gateway@$USER -f
```

(`deploy/llm-gateway@.service` is a template unit: its `%i` instance is the user that owns
the checkout at `/home/<user>/llm-gateway`.)

The unit is `Restart=always` with `StartLimitIntervalSec=0`, so a boot storm cannot leave
it permanently failed. It binds the Tailscale address when one exists and keeps checking,
so a `tailscaled` that comes up late still gets a listener.

Request log: `~/.local/state/llm-gateway/requests.jsonl` — one JSON object per request.

## Design notes

- **No provider list in the code.** Baking prices or provider names into the proxy makes
  it stale within hours and impossible to audit. The routing file is the only source of
  truth, and it is re-read on mtime+size change.
- **Streaming uses chunked encoding.** A 1.1 response with neither `Content-Length` nor
  `Transfer-Encoding` hangs strict clients; keep-alive frames are forwarded as SSE
  comments.
- **Binding is supervised, not one-shot.** At boot the tailnet address may not exist yet;
  a one-shot bind silently leaves the gateway loopback-only until someone restarts it.
- **Client disconnects are logged as 499, not errors.** Hanging up mid-stream is normal.

## Limitations

- OpenRouter only (the upstream host is a constant; add an env var if you need another).
- No response caching, rate limiting, or multi-key rotation.
- One process per port; scaling out means putting a real reverse proxy in front.

## License

MIT — see [LICENSE](LICENSE).
