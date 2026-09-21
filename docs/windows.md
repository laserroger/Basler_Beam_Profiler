# Windows standalone installer

The new Windows packaging is intended to produce one offline
`BeamProfiler-Windows-x64-Setup.exe` for a Windows x64 computer without Python
or camera software. It includes the application, matching PySpin binding,
Spinnaker DLLs and GenTL transport, and the signed vendor installer for runtime
prerequisites and USB camera drivers. Windows administrator approval is required
when installing drivers. Users should not need separate dependency downloads.

The older `pylon_camera.exe` releases include Basler support but lack Windows FLIR
runtime/binding packaging.

## Building

On a Windows x64 build machine, install Python 3.12 x64 and Inno Setup 6. Obtain
the Windows x64 **full/offline** Spinnaker 4.x installer and matching
`spinnaker_python-…-cp312-cp312-win_amd64.whl` from Teledyne. The web installer
is unsuitable because the finished setup must work offline.

From PowerShell:

```powershell
./tools/build_windows.ps1 -SpinnakerInstaller C:/Downloads/SpinnakerSDK_FULL_VERSION_x64.exe -PySpinWheel C:/Downloads/spinnaker_python-VERSION-cp312-cp312-win_amd64.whl
```

The script checks the vendor signature, bundles
the native files and license notices, verifies both camera drivers and a short
simulator run, then compiles the offline installer. Match the SDK and wheel
versions. Do not use the unrelated `pyspin` package on PyPI.

Before release, test the installer on a separate clean Windows machine with no
Spinnaker SDK, Python, or vendor paths; initialize both drivers, open settings,
and acquire frames from the connected FLIR. A simulator or driver-load check
alone is not a live-camera acceptance test.

The vendor install features follow [Teledyne's distribution instructions](https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/distributing-spinnaker-applications/).
