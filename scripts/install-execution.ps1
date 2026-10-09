param([switch]$ConfirmExecutionMachine, [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
if (-not $ConfirmExecutionMachine) {
    throw 'Run only on the authorized execution host with -ConfirmExecutionMachine.'
}
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $repoRoot
& $Python -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ is required"'
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw 'Install a standalone FFmpeg build and add its bin folder to PATH. Do not use SubAI runtime.'
}
& $Python -m venv .venv-vnle
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$runtimePython = Join-Path $repoRoot '.venv-vnle\Scripts\python.exe'
& $runtimePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
& $runtimePython -m pip install -e '.[analysis,cuda]'
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
Write-Host 'Environment installed. Model preparation is explicit; see docs/PROTOTYPE.md.'
