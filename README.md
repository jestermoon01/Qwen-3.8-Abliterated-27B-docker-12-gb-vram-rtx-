# Local Huihui Qwen3.8 27B on Docker

Run a quantized 27B model locally with a browser chat UI, an OpenAI-compatible
API, automatic context compression, and persistent Mem0/Qdrant memory.
The default is tuned for an NVIDIA RTX 5070 with 12 GB VRAM. It uses llama.cpp
CUDA with IQ3 weights, full GPU offload, and native MTP speculative decoding.

Compose downloads the pinned model and verifies its SHA256 automatically.
No personal files, machine-specific absolute paths, model weights, or credentials
are required from the original computer.

## Supported machines

- Windows x86-64 with Docker Desktop using WSL2 and NVIDIA GPU support, or
  Linux x86-64 with Docker Engine and NVIDIA Container Toolkit configured.
- A compatible NVIDIA GPU and current driver. The tested default needs about
  12 GB VRAM and leaves little headroom; GPU architecture, display usage, context,
  and backend compatibility affect whether it fits. Only the RTX 5070 machine
  has been tested with real inference.
- Docker Compose **2.30 or newer** (GPU support and Compose override tags).
- Plan for at least 32 GB host RAM and 30 GB free disk for weights, images,
  and caches. The original machine has 64 GB RAM. Allow Docker/WSL adequate memory.
- Internet for initial Docker images, the **11.02 GB** GGUF download, and the
  small embedding model. Later inference and memory run locally using cached files.

This CUDA deployment does **not** run unchanged on macOS/Apple Silicon, AMD-only,
or CPU-only devices. Docker does not make GPU requirements universal.

## Start

```sh
git clone https://github.com/jestermoon01/Qwen-3.8-Abliterated-27B-docker-12-gb-vram-rtx-.git
cd Qwen-3.8-Abliterated-27B-docker-12-gb-vram-rtx-
docker compose up -d --build
```

No `.env` file is needed for the defaults. To customize, copy `.env.example` to
`.env` before starting. Convenience scripts are `scripts/setup.ps1` for
PowerShell and `sh scripts/setup.sh` for Linux.

First startup may take a long time. Compose runs the model downloader to
completion before starting inference, and waits for inference readiness before
starting the gateway. Download progress appears in the logs:

```sh
docker compose logs -f model-download inference gateway
docker compose ps -a
```

The downloader should exit with code 0; inference and gateway should be healthy.
Then open **http://localhost:8001/**. The API base URL is
**http://localhost:8001/v1**, model **huihui-qwen3.8-27b**.
There is no API key. If `API_PORT` is changed, use that port instead.

Verify readiness and send a request (use `curl.exe` in Windows PowerShell):

```sh
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

```python
import json
from urllib.request import Request, urlopen

payload = {
    "model": "huihui-qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain Docker volumes briefly."}],
    "max_tokens": 128,
}
request = Request("http://localhost:8001/v1/chat/completions",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

The API binds to `127.0.0.1`; inference and Qdrant have no host ports.
Each device runs its own copy. This does not expose the API to other computers.
See [agent integration](integrations/README.md) to connect Hermes, ARTEX, or PentAGI.

## Model and persistence

`model-manifest.json` pins the upstream repository, revision, filename, size,
and SHA256. The downloader uses only Python's standard library, retries failed
downloads, checks the completed file, and atomically publishes it. Interrupted
downloads restart from the beginning on retry. Incomplete files are never loaded
by inference. Cached files are verified again on subsequent downloader runs.

Named volumes hold model weights (`models`), embeddings and memory history
(`memory`), and vectors (`qdrant`). Ordinary restarts and `docker compose down`
retain them. **`docker compose down -v` deletes all these volumes**, including
the downloaded weights and memories.

To reuse a complete GGUF already on disk, set `MODEL_PATH` to its absolute path
in `.env`, then use the local-file override on every Compose command:

```sh
docker compose -f compose.yaml -f compose.local-model.yaml up -d --build
```

This skips downloading and mounts the file read-only. Use the pinned IQ3 model
for the tested settings; a different model can require different context, cache,
GPU, and speculative-decoding settings. Renaming `MODEL_ALIAS` only changes the
API name, not the weights. For a different automatically downloaded checkpoint,
update all model-manifest identity and checksum fields together.

## Presets and resources

Defaults: one active inference slot, 8,192 tokens including output, eight CPU
threads, Q4 KV cache, all decoder layers on GPU, and MTP enabled. Additional
requests queue; the gateway admits up to eight in-flight requests, then returns
429. The model is shared by all agents.

To use two slots with 16,384 tokens each, copy `profiles/balanced-iq3.env` to
`.env`, then recreate services. This disables MTP to leave room for context.
Use `profiles/fast-iq3-mtp.env` to restore the default preset.

PowerShell:

```powershell
Copy-Item profiles/balanced-iq3.env .env -Force
docker compose up -d --build
```

Linux:

```sh
cp profiles/balanced-iq3.env .env
docker compose up -d --build
```

Keep `CONTEXT_PER_SLOT = TOTAL_CONTEXT / PARALLEL_SLOTS`. Startup rejects a
gateway/backend context mismatch. Lower `GPU_LAYERS` if the model runs out of
VRAM; CPU offload needs more RAM and reduces speed. Stop other GPU-heavy work.
Keep the measured defaults unless you plan to benchmark changes.

`compose.long.yaml` is an experimental 65,536-token-per-slot, two-slot override
using CPU KV cache and partial GPU offload. It needs substantially more RAM and
is not validated on every machine. Use both files consistently:

```sh
docker compose -f compose.yaml -f compose.long.yaml up -d --build
```

Original-machine short warm-generation measurements were about 65 tokens/s for
the fast preset and 43 tokens/s for one request on the balanced preset. These
are hardware-specific observations, not performance guarantees. Reasoning and
tool-use errors remain; quantization and longer context can affect quality.

## Context and memory

The gateway counts tokens using the backend's real chat template, reserves
output space, and summarizes older turns when needed. Compression is lossy;
recent tool-call IDs, ordering, arguments, and system/developer instructions are
preserved. Large tool results can be summarized with a lossy-summary marker;
oversized instructions, user requests, or call arguments can still return 413. Text, streaming,
and OpenAI-style function tools are supported; clients execute the actual tools.
Images and audio are not supported. This is a model backend, not an agent runner.

Send `X-User-ID` to enable semantic memory and `X-Agent-ID` to select an agent
namespace. Omitting the user header disables automatic memory for that request.
Successful user turns are saved verbatim through a durable outbox. Mem0 uses
local FastEmbed `BAAI/bge-small-en-v1.5` embeddings and Qdrant. Recall is
eventually consistent and the embedding model is English-oriented.

Memory routes: `POST /memory` with `{"text":"..."}`, `GET /memory/search?q=...`,
and `DELETE /memory`, with the same namespace headers. Headers organize memory;
they are not tenant authentication. Use this unauthenticated service only in
your trusted local environment.

## Operations and troubleshooting

```sh
docker compose logs --tail 100 gateway inference model-download
docker compose exec gateway python -m unittest discover -s tests -p 'test_*.py' -v
docker compose exec gateway python tests/live.py
docker compose stop
docker compose start
```

If Compose cannot select a GPU driver, check `nvidia-smi`, Docker GPU access,
and the NVIDIA Container Toolkit/WSL2 setup. If the downloader fails, inspect
its logs and disk space, then rerun `docker compose up -d --build`. If inference
is healthy but gateway startup fails, inspect embedding downloads, Qdrant,
network access, and available RAM. Do not increase context without checking VRAM.

`python benchmarks/run_hard.py` records a custom reasoning/tool/memory benchmark
against the running API. It is not a standard model quality score. More extensive
live checks send real inference requests and can take several minutes.

The CUDA backend, Qdrant, base Python image, model revision, and Python dependency
versions are pinned. Fresh machines still need compatible drivers and download
access. Upstream model and dependency licenses apply; this repository does not
redistribute weights or claim ownership of them.
