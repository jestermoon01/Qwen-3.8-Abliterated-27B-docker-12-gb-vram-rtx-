#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker compose up -d --build
echo 'Follow startup: docker compose logs -f model-download inference gateway'
echo 'When gateway is healthy: http://localhost:8001/ (or your configured API_PORT).'
