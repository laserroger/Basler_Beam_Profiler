<# Build an application-only installer with bundled camera runtime libraries.
   Requires Python 3.12 x64, Inno Setup 6, and the vendor PySpin wheel.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PySpinWheel,
    [string]$Iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    [switch]$SpotTracing,
    [switch]$SkipChecks
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Run this build on Windows x64.' }
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$wheel = (Resolve-Path $PySpinWheel).Path
if ((Split-Path $wheel -Leaf) -notmatch '^spinnaker_python-.*-cp312-cp312-win_amd64\.whl$') {
    throw 'Select the vendor spinnaker_python wheel for CPython 3.12, Windows x64.'
}
if (-not (Test-Path $Iscc)) { throw 'Install Inno Setup 6 on the build machine.' }
python -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 x64 is required on the build machine.' }

python -m pip install -r requirements-windows-build.txt $wheel
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
$licenses = Join-Path $repo '.vendor/windows/licenses'
if (-not (Test-Path $licenses)) { throw 'Missing vendor license notices.' }
$previousSpotBuild = $env:BEAM_SPOT_TRACING_BUILD
try {
    $env:BEAM_SPOT_TRACING_BUILD = if ($SpotTracing) { '1' } else { '0' }
    python -m PyInstaller --noconfirm pylon_camera_windows.spec
} finally {
    $env:BEAM_SPOT_TRACING_BUILD = $previousSpotBuild
}
if ($LASTEXITCODE -ne 0) { throw 'Application build failed.' }
$app = Join-Path $repo 'dist/pylon_camera'
if (-not $SkipChecks) {
    # Clear SDK search variables: verify the app resolves its packaged libraries.
    Remove-Item Env:SPINNAKER_GENTL64_CTI -ErrorAction SilentlyContinue
    Remove-Item Env:GENICAM_GENTL64_PATH -ErrorAction SilentlyContinue
    $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notmatch '(?i)Spinnaker' }) -join ';'
    & "$app/pylon_camera.exe" --check-drivers
    if ($LASTEXITCODE -ne 0) { throw 'Bundled camera-driver initialization failed.' }
    & "$app/pylon_camera.exe" --sim --frames 3
    if ($LASTEXITCODE -ne 0) { throw 'Packaged app smoke test failed.' }
}
$defines = @("/DAppSource=$app", "/DOutputPath=$repo\dist")
if ($SpotTracing) { $defines += '/DSpotTracing=1' }
& $Iscc @defines "$repo\packaging\windows\BeamProfiler.iss"
if ($LASTEXITCODE -ne 0) { throw 'Offline installer build failed.' }
$outputName = if ($SpotTracing) { 'BeamProfiler-SpotTracing-Windows-x64-Setup.exe' } else { 'BeamProfiler-Windows-x64-Setup.exe' }
Write-Output "Built $repo\dist\$outputName"
