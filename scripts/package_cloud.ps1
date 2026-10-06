param([string]$Output = '.cache\cn2vi-cloud.bundle', [switch]$BundleOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
$status = & git status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect repository state.' }
if ($status) { throw 'Commit the prepared changes before packaging cloud code.' }
$branch = & git symbolic-ref --short HEAD
if ($LASTEXITCODE -ne 0) { throw 'Packaging requires a committed local branch.' }
if (-not $BundleOnly) {
    Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
    try {
        & npm ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency install failed.' }
        & npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
    if (& git status --porcelain) { throw 'Repository changed during frontend build; commit before packaging.' }
}
$bundlePath = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
} else { [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Output)) }
if (-not $bundlePath.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Cloud bundle must be saved inside this workspace.'
}
New-Item -ItemType Directory -Force -Path (Split-Path $bundlePath -Parent) | Out-Null
& git bundle create $bundlePath $branch
if ($LASTEXITCODE -ne 0) { throw 'Cloud bundle creation failed.' }
& git bundle verify $bundlePath
if ($LASTEXITCODE -ne 0) { throw 'Cloud bundle verification failed.' }
$digest = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath ($bundlePath + '.sha256') -Value ($digest + '  ' + (Split-Path $bundlePath -Leaf)) -Encoding Ascii
Write-Output "Cloud code bundle ready: $bundlePath"
Write-Output "SHA-256: $digest"
if (-not $BundleOnly) {
    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Install the core Python environment first.' }
    $commit = & git rev-parse HEAD
    $installer = Join-Path (Split-Path $bundlePath -Parent) 'cn2vi-cloud-setup.run'
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $projectRoot 'src'
        & $python -m autodub.cloud_package --project-root $projectRoot --bundle $bundlePath --commit $commit --output $installer
        if ($LASTEXITCODE -ne 0) { throw 'Cloud installer packaging failed.' }
    } finally { $env:PYTHONPATH = $previousPythonPath }
    Write-Output "Standalone cloud setup: $installer"
}
