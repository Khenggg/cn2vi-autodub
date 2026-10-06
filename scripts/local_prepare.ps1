param(
    [switch]$TtsCpu,
    [string]$BasePython = 'python',
    [string]$Uv = 'uv'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot

$baseCommand = Get-Command $BasePython -ErrorAction Stop
$basePythonPath = $baseCommand.Source
$pythonVersion = & $basePythonPath -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne '3.12') {
    throw 'A system Python 3.12 interpreter is required; no interpreter is downloaded automatically.'
}
$uvCommand = Get-Command $Uv -ErrorAction Stop
$uvPath = $uvCommand.Source
$cacheRoot = Join-Path $projectRoot '.cache'
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
$sourceRoot = Join-Path $projectRoot 'src'

function Install-Profile([string]$Name, [string]$LockFile, [string[]]$RequiredModules) {
    $environment = Join-Path $cacheRoot $Name
    $python = Join-Path $environment 'Scripts\python.exe'
    $lockPath = Join-Path $projectRoot $LockFile
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) { throw "Missing hashed profile: $LockFile" }

    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        & $uvPath venv --no-python-downloads --python $basePythonPath $environment
        if ($LASTEXITCODE -ne 0) { throw "Could not create $Name environment" }
    }
    & $uvPath pip sync --python $python --require-hashes $lockPath
    if ($LASTEXITCODE -ne 0) { throw "Could not install hashed profile $LockFile" }
    & $uvPath pip check --python $python
    if ($LASTEXITCODE -ne 0) { throw "$Name profile has dependency conflicts" }

    $sitePackages = & $python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
    if ($LASTEXITCODE -ne 0) { throw "Could not locate $Name site-packages" }
    $pthFile = Join-Path $sitePackages 'cn2vi-autodub-src.pth'
    Set-Content -LiteralPath $pthFile -Value $sourceRoot -Encoding Ascii -NoNewline
    $moduleList = ($RequiredModules | ForEach-Object { "'$_'" }) -join ','
    $verifyCode = "import importlib,importlib.util,sys; exec('def _ok(name):\n try: importlib.import_module(name); return True\n except Exception: return False'); required=[$moduleList]; missing=[name for name in required if importlib.util.find_spec(name) is None]; failures=[name for name in required if name not in missing and not _ok(name)]; print('missing modules:', ','.join(missing)); print('failed imports:', ','.join(failures)); sys.exit(bool(missing or failures))"
    & $python -c $verifyCode
    if ($LASTEXITCODE -ne 0) { throw "$Name profile is missing a required module" }
    if ($Name -eq 'vision-env') {
        $hasFastApi = & $python -c 'import importlib.util; print(importlib.util.find_spec("fastapi") is not None)'
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect $Name environment" }
        if ($hasFastApi.Trim() -eq 'True') { throw 'Vision model environment unexpectedly contains FastAPI' }
    }
    Write-Output "$Name ready: $python"
}

Install-Profile 'vision-env' 'requirements\bench-vision.txt' @('psutil', 'numpy', 'PIL', 'onnxruntime', 'rapidocr')
if ($TtsCpu) {
    Install-Profile 'tts-cpu-env' 'requirements\bench-tts-cpu.txt' @('psutil', 'numpy', 'onnxruntime', 'soundfile', 'vieneu')
}
Write-Output 'Model environments prepared from hashed locks. No model assets or GPU packages were fetched.'
