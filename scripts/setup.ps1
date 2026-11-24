$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { throw 'Docker deployment failed. Check docker compose logs.' }
    Write-Host 'Downloading/loading may take a while. Follow: docker compose logs -f model-download inference gateway'
    Write-Host 'When gateway is healthy: http://localhost:8001/ (or your configured API_PORT).'
} finally {
    Pop-Location
}
