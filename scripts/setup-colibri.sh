#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker compose stop inference gateway
docker compose -f compose.colibri.yaml run --rm model-download python /setup/download_model.py --manifest /setup/model-manifest.json --destination /weights/model.gguf --interactive
docker compose -f compose.colibri.yaml up -d --build --wait --wait-timeout 600
printf '%s\n' 'Colibri IQ2: http://localhost:8002/v1 (default port). IQ3 is stopped.'
