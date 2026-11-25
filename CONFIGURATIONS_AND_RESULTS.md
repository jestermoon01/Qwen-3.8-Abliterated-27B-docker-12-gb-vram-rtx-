# IQ3 for Hermes and optional Colibri IQ2

## Recommendation

Keep **llama.cpp + IQ3 as the default for Hermes**. It preserves the current
weight precision and working OpenAI-compatible connection. For an agent,
tool selection and answer quality matter alongside generation speed. The speed
comparison did not establish that IQ2 is equally reliable for agent tasks.

Colibri is an inference engine; IQ3 and IQ2 are weight quantizations. Both
configurations below use the Huihui Qwen3.8 27B checkpoint family. The Colibri
experiment uses a different, lower-precision GGUF and is optional.

| Setting | Recommended IQ3 | Optional Colibri IQ2 |
|---|---|---|
| Engine | llama.cpp CUDA, pinned image | Colibri CUDA, pinned source revision |
| Quantization | UD-IQ3_XXS | UD-IQ2_S-MTP file; MTP disabled |
| Weights | 11,021,122,464 bytes | 8,775,408,544 bytes |
| Context | 32,768 total tokens, including output | 32,768 total tokens, including output |
| Active slots | 1 | 1 |
| GPU placement | All decoder layers | All 64 layers and output head on CUDA |
| Cache | Q4 K/V, flash attention | Float16 K/V |
| Speculation | Off | MTP and prompt lookup off |
| CPU threads | 8 | 8 |
| Prompt microbatch | 128 | 128 CUDA chain rows |
| Thinking default | On | Model default; tests explicitly disabled it |
| Host API | `http://localhost:8001/v1` | `http://localhost:8002/v1` |
| Model ID | `huihui-qwen3.8-27b` | `huihui-qwen3.8-27b` |

Only run one GPU inference backend at a time on a 12 GB card.

## Run IQ3 and keep Hermes connected

For a fresh checkout, follow [README.md](README.md):

```powershell
# Windows: download/verify IQ3, build the UI proxy, start the services.
.\scripts\setup.ps1
```

```sh
# Linux equivalent.
sh scripts/setup.sh
```

The defaults already select the recommended 32K configuration.
`profiles/iq3-hermes.env` contains an explicit example. Keep an existing `.env`
when it already contains your correct connection settings. To rebuild in place:

```sh
docker compose up -d --build --wait --wait-timeout 600
```

For a host-mounted GGUF deployment, retain the `compose.local-model.yaml`
override on subsequent commands, as described in the README.

Hermes connection values:

| Setting | Value |
|---|---|
| Provider protocol | OpenAI-compatible Chat Completions |
| Base URL, Hermes on the same host | `http://localhost:8001/v1` |
| Base URL, Hermes attached to `qwen-local_default` in Docker | `http://gateway:8000/v1` |
| Model | `huihui-qwen3.8-27b` |
| API key | None; `local-not-required` if the client requires a nonempty value |
| Context budget | 32,768, including system prompt, tools, history and output |
| Reply budget | 2,048 normally; 4,096 was tested below |

Keep existing Hermes settings if they already use this endpoint. Rebuilding
keeps the Compose project name, host port, `gateway` network alias, model ID,
and model volume. Container IP addresses may change; use the URL/alias above.

The public package is a lightweight UI/API proxy. The original workstation's
deployment also has a memory/compression gateway and Qdrant; those local app
services and credentials are not bundled here. Rebuild an existing workstation
stack from its own Compose directory rather than replacing it with this package.

## Run the optional Colibri IQ2 deployment

The IQ2 weights are downloaded from a pinned revision and verified against
`model-manifest.iq2.json`. Colibri is built from source commit
`e479104ae01bdc40ce81e675299b1d230891b321`, using CUDA 12.8.1 and `sm_120`
for this RTX 5070. Its model importer links the existing GGUF without a second
weight copy or another quantization pass. Model weights and import/cache state
use separate named volumes, independent of the IQ3 deployment.

Allow about 60 GB of free disk for both sets of weights, CUDA images and build
cache. First compilation can take several minutes. The tests used 64 GB host
RAM; the Colibri server is configured with a 24 GB RAM budget.

```powershell
# Windows: stops IQ3, verifies/downloads IQ2, builds and starts Colibri.
.\scripts\setup-colibri.ps1
```

```sh
# Linux equivalent.
sh scripts/setup-colibri.sh
```

For the current public weights, a noninteractive alternative is:

```sh
docker compose stop inference gateway
docker compose -f compose.colibri.yaml up -d --build --wait --wait-timeout 600
```

Colibri's separate API is `http://localhost:8002/v1`. The model ID is unchanged,
but this does not automatically redirect an existing Hermes client from port
8001. The recommended Hermes deployment stays on IQ3. Colibri does not expose
all llama.cpp-specific tokenizer/template routes used by the original local
memory gateway, so it is not a transparent replacement for that gateway.

To reuse the exact IQ2 GGUF already on disk, verify its SHA256 against the IQ2
manifest and set `COLIBRI_MODEL_PATH` to its absolute path. This avoids another
download and mounts it read-only:

```powershell
$env:COLIBRI_MODEL_PATH = 'C:/models/huihui-27b-iq2_s-mtp.gguf'
docker compose stop inference gateway
docker compose -f compose.colibri.yaml -f compose.colibri.local-model.yaml up -d --build --wait --wait-timeout 600
```

Use both Colibri files on later commands for that local-file deployment.

Restore IQ3 and the original agent URL:

```sh
docker compose -f compose.colibri.yaml stop
docker compose up -d --wait --wait-timeout 600
```

Ordinary stops/restarts preserve volumes. Do not use `down -v` if you want to
retain downloaded weights and cached state.

## Measured results: 2026-11-25

Hardware: RTX 5070, 12,227 MiB VRAM, NVIDIA driver 610.88; Intel i7-14700K;
64 GB host RAM. One active inference request, thinking explicitly disabled.
Each completed generation below produced exactly 4,096 output tokens and ended
at the requested length limit. Codes at both ends of the long input were recalled.

| Configuration | Input tokens | Output tokens | First token | Observed generation | Total request |
|---|---:|---:|---:|---:|---:|
| llama.cpp IQ3, long input | 24,561 | 4,096 | 22.34 s | 23.65 tokens/s | 195.47 s |
| Colibri CUDA IQ3, short input | 114 | 4,096 | 0.38 s | 34.88 tokens/s | 117.78 s |
| Colibri CUDA IQ2, long input, MTP off | 24,504 | 4,096 | 58.82 s | 33.72 tokens/s | 180.27 s |

Generation rate is `(completion_tokens - 1) / (last_content_time - first_content_time)`
observed over SSE. Total request time also includes prompt processing. The original
IQ3 gateway adds a system instruction, explaining the 57-token input difference
for the same long user prompt. The public lightweight proxy does not add it.

IQ2 generation was about 43% faster than the long-input IQ3 baseline, but the
whole request was only about 8% faster because prompt ingestion took longer.
These measurements do not demonstrate 40 tokens/s with a filled 32K context,
nor equal agent correctness. IQ2 reduces precision and can change answers.
Boundary recall is a limited check, not a general quality benchmark.

Other configurations were stopped rather than counted as completed benchmarks:

- IQ3 Colibri CUDA with long input: first token at 112.44 s, then very slow
  generation near the VRAM limit.
- IQ3 Vulkan with default placement: KV cache split into RAM and one-row prompt
  batches. Stopped before the first token.
- IQ3 Vulkan with 60 GPU layers and four CPU layers: first token at 468.61 s;
  boundary recall passed, but the reply was not allowed to run to 4K tokens.
- IQ2 with three-draft and one-draft MTP: very slow generation near the VRAM
  limit; stopped. Plain IQ2 decoding was the successful long-output configuration.

CUDA used the pinned local source build. Vulkan used the checksum-verified
official Windows v2.2.0 release, so those trials also differ in platform and
placement. Results are specific to these runs, not hardware ceilings.

## Reproduce the streamed test

Python 3 standard library only; run from the repository root. Load and warm one
backend at a time before comparing. The request disables thinking independently
of the server default; system/tool prompts and enabled reasoning change latency.

```sh
python benchmarks/stream_hardware.py --label iq3-long --lines 1500 --tokens 4096

# After switching to Colibri IQ2:
python benchmarks/stream_hardware.py --label colibri-iq2-long --url http://localhost:8002 --weights huihui-27b-iq2_s-mtp.gguf --lines 1500 --tokens 4096
```

Reports are saved to `benchmarks/results/stream-<label>.json`. They include the
exact request, generated text, actual token usage, first-token timing, finish
reason, rate, and boundary recall. A first-content-token deadline and a total
request deadline prevent an SSE heartbeat from hiding a stalled request.

The recorded synthetic requests/replies and summary are in
[benchmarks/data/comparison-2026-11-25.json](benchmarks/data/comparison-2026-11-25.json).
No model weights, private agent settings, or workstation credentials are included.

## Existing workstation rebuild verification

On 2026-11-25 the original local IQ3 inference and gateway containers were
rebuilt/recreated from their existing Compose project. The host binding stayed
`127.0.0.1:8001`, the internal `gateway:8000` alias and `qwen-local_default`
network were retained, and mount/volume names matched the before-rebuild
snapshot. The memory/Qdrant volumes and agent configuration were preserved.

The existing Hermes CLI completed a one-shot reply through its configured
connection without configuration changes. A forced function-call API probe
passed, with 366 prompt tokens counted by both gateway and backend. All five
local context-budget tests passed. These are connectivity/protocol checks,
not a benchmark of Hermes task-solving accuracy or an IQ2 quality comparison.
