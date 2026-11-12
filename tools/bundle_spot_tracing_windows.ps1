<# One-time Windows x64 bundle for the live-spot-diameters branch.
   Packages runtime libraries only. Does not run the app or tests.
#>
[CmdletBinding()]
param(
    [string]$Iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Run this bundling tool on Windows x64.' }
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    python -m tools.fetch_windows_vendor
    if ($LASTEXITCODE -ne 0) { throw 'Runtime-library download failed.' }
    $manifest = Get-Content 'external_libs/flir/windows-x64/manifest.json' -Raw | ConvertFrom-Json
    $wheels = @($manifest.files.PSObject.Properties.Name | Where-Object { $_ -like '*.whl' })
    if ($wheels.Count -ne 1) { throw 'Expected exactly one pinned Windows PySpin wheel.' }
    $wheel = Join-Path '.vendor/windows' $wheels[0]
    & "$PSScriptRoot/build_windows.ps1" -PySpinWheel $wheel -Iscc $Iscc -SpotTracing -SkipChecks
    if ($LASTEXITCODE -ne 0) { throw 'Spot-tracing bundle failed.' }
    Write-Output 'Ready: dist/BeamProfiler-SpotTracing-Windows-x64-Setup.exe'
} finally {
    Pop-Location
}
