param([string]$Output = '.cache\cn2vi-cloud.bundle')
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
$status = & git status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect repository state.' }
if ($status) { throw 'Commit the prepared changes before packaging cloud code.' }
$branch = & git symbolic-ref --short HEAD
if ($LASTEXITCODE -ne 0) { throw 'Packaging requires a committed local branch.' }
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
