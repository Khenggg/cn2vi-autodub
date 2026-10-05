$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
$testRunRoot = Join-Path $projectRoot ('.cache\tests-' + [guid]::NewGuid().ToString('N'))
# A fresh workspace-local temp directory avoids shared Windows temp ACL issues.
& .\.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp=$testRunRoot -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\ruff.exe check src tests benchmarks
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
try {
    & npm run build
    $checkExit = $LASTEXITCODE
} finally { Pop-Location }
exit $checkExit
