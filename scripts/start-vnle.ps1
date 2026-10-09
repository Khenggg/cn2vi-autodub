param([int]$Port = 18083)
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $repoRoot
$runtimePython = Join-Path $repoRoot '.venv-vnle\Scripts\python.exe'
$manifest = Join-Path $repoRoot 'models\ppocr-v6-small\manifest.json'
if (-not (Test-Path -LiteralPath $runtimePython)) { throw 'Install the VNLE environment first.' }
if (-not (Test-Path -LiteralPath $manifest)) { throw 'Prepare the pinned OCR models first.' }
& $runtimePython -m vnle serve --port $Port --data-root (Join-Path $repoRoot 'data\vnle') --config (Join-Path $repoRoot 'config\prototype.cuda.json') --model-manifest $manifest --enable-analysis
exit $LASTEXITCODE
