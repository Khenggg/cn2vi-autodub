param([int]$Port = 18083, [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# UI-only: no dependency install, no model download and analysis remains disabled.
$env:PYTHONPATH = Join-Path $repoRoot 'src'
& $Python -m vnle serve --port $Port --data-root (Join-Path $repoRoot 'data\vnle')
exit $LASTEXITCODE
