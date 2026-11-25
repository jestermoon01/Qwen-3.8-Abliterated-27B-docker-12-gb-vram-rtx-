$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    docker compose stop inference gateway
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the IQ3 GPU service.' }
    docker compose -f compose.colibri.yaml run --rm model-download python /setup/download_model.py --manifest /setup/model-manifest.json --destination /weights/model.gguf --interactive
    if ($LASTEXITCODE -ne 0) { throw 'IQ2 model download or verification failed.' }
    docker compose -f compose.colibri.yaml up -d --build --wait --wait-timeout 600
    if ($LASTEXITCODE -ne 0) { throw 'Colibri startup failed; see docker compose -f compose.colibri.yaml logs.' }
    Write-Host 'Colibri IQ2: http://localhost:8002/v1 (default port). IQ3 is stopped.'
} finally { Pop-Location }
