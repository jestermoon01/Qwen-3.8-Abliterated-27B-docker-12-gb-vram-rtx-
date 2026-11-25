# Qwen 27B: local inference and chat UI

Deploy Huihui Qwen3.8 27B IQ3 with llama.cpp CUDA, a browser chat UI, and an
OpenAI-compatible API for agent clients. IQ3 is the recommended default for
Hermes. An optional Colibri CUDA IQ2 deployment is included separately.

See **[CONFIGURATIONS_AND_RESULTS.md](CONFIGURATIONS_AND_RESULTS.md)** for both
setups, switching instructions, Hermes connection settings and measured results.

The defaults match the inspected running Docker inference container: the same
IQ3 GGUF, pinned llama.cpp CUDA image, 32K context, one slot, thinking enabled,
and speculative decoding disabled. The downloaded file must match SHA256
`e86259c841888b06fd84750963a0263947c9aae3c3029b52cbb3faa34a554827`.

## Requirements

- Windows x86-64 with Docker Desktop/WSL2 and NVIDIA GPU access, or Linux
  x86-64 with Docker Engine and NVIDIA Container Toolkit.
- Docker Compose 2.30 or newer and a compatible NVIDIA driver/GPU.
- Defaults are tuned for an RTX 5070 with **12 GB VRAM**. Plan for at least
  **32 GB host RAM** and **30 GB free disk**. Other GPUs have not been validated.
  macOS, AMD-only, and CPU-only devices need a different inference backend.
- Internet for initial images and the **11.02 GB** model download.

## Clone and start

```sh
git clone https://github.com/jestermoon01/Qwen-3.8-Abliterated-27B-docker-12-gb-vram-rtx-.git
cd Qwen-3.8-Abliterated-27B-docker-12-gb-vram-rtx-
```

Windows PowerShell:

```powershell
.\scripts\setup.ps1
```

Linux:

```sh
sh scripts/setup.sh
```

Setup downloads and verifies the pinned model, then builds and starts inference
and the UI. **This public model currently needs no Hugging Face token**, verified
with an unauthenticated request. If Hugging Face returns 401/403, setup pauses
for a **hidden read token** and then retries. Obtain a token from
https://huggingface.co/settings/tokens and approve any model access terms first.
A token cannot bypass access approval. Blank input cancels.

The token stays in the downloader process only. It is never saved, committed,
passed as a build argument, or placed in an image. **No HF token is needed during
Docker build**: model downloading happens at setup/runtime in a persistent volume.

For deployment without interactive prompts:

```sh
docker compose up -d --build
```

This works for the current public model. If access is restricted, it fails with
instructions to run setup. Detached Compose cannot accept a token prompt; use
the setup script when interactive authentication is needed.

Follow startup until inference and gateway are healthy:

```sh
docker compose logs -f model-download inference gateway
docker compose ps -a
```

Open **http://localhost:8001/**. Chat history stays in browser memory until refresh.
Start a new chat when you reach the context limit. This public package does not
bundle agent frameworks, memory services, databases, or automatic compression.
Benchmark scripts and recorded comparison data are included for reproduction.

## Attach an agent

Configure the agent's OpenAI-compatible provider:

| Setting | Value |
|---|---|
| Base URL on the same host | `http://localhost:8001/v1` |
| Model | `huihui-qwen3.8-27b` |
| API key | None; use `local-not-required` if the client requires a value |

For an agent in Docker, attach its container to `qwen-local_default` and use
`http://gateway:8000/v1`. Inside a container, `localhost` refers to that container.
The API proxies llama.cpp's `/v1/*` endpoints, including chat completions and
streaming. Tool calls are returned to the agent, which executes its own tools.
Clients requiring another protocol or unsupported multimodal features need an adapter.

The API binds to loopback and has no authentication. It is intended for trusted
local use. Each laptop runs its own deployment; other computers cannot connect
by default.

## Configuration and model storage

Copy `.env.example` to `.env` to change the port, CPU threads, GPU layers, or
context. Defaults: full GPU offload, eight CPU threads, one active slot, **32768
tokens including output**, Q4 KV cache, thinking enabled, speculative decoding
disabled, and microbatch size 128. Keep
`CONTEXT_PER_SLOT = TOTAL_CONTEXT / PARALLEL_SLOTS`. Extra agents share the model.

If VRAM is insufficient, stop other GPU workloads or lower `GPU_LAYERS` for
slower CPU offload. Resource requirements vary with settings.

`model-manifest.json` pins the upstream repository, revision, filename, size,
and SHA256. Weights are verified before inference starts. Failed downloads
retry from the beginning; cached weights are verified and reused. The named
`models` volume survives restart and `docker compose down`.
**`docker compose down -v` deletes the downloaded model.**

To reuse a compatible GGUF already on disk, set `MODEL_PATH` to its absolute
path in `.env` and use this command instead of setup:

```sh
docker compose -f compose.yaml -f compose.local-model.yaml up -d --build
```

Use both files on subsequent Compose commands. This skips downloading and
mounts the file read-only. Other checkpoints may need different inference settings.

## Operations

```sh
docker compose logs --tail 100 model-download inference gateway
docker compose stop
docker compose start
docker compose down
```

Readiness: `http://localhost:8001/health`. Models: `http://localhost:8001/v1/models`.
If GPU startup fails, check `nvidia-smi`, Docker GPU access, and free VRAM. For
model download failures, check network access, disk space, and Hugging Face
permissions, then rerun setup. GPU compatibility and performance on other laptops
are not guaranteed. Upstream model and dependency licenses apply.
