# Windows application-only installer

`BeamProfiler-Windows-x64-Setup.exe` includes the application, Python, matching
PySpin binding, Spinnaker DLLs and GenTL transport, and Basler runtime support.
It uses the camera drivers already installed on the destination computer.
No Spinnaker SDK installer or camera-driver installer is downloaded for the
build, embedded in the package, or executed by Setup. Installing or uninstalling
Beam Profiler leaves existing camera software and drivers untouched.

## Building

On Windows x64, use Python 3.12 x64 and Inno Setup 6. Fetch the pinned PySpin
wheel and license notices with `python -m tools.fetch_windows_vendor`, then:

```powershell
./tools/build_windows.ps1 -PySpinWheel .vendor/windows/spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl
```

The build collects the wheel's matching native runtime libraries into the app
and compiles the application-only installer. The GitHub workflow uses the same
build path. Its installer verification expects no system SDK to be installed
by Setup; runtime-library checks use the app's bundled libraries.

## Spot-tracing branch package

Run `./tools/bundle_spot_tracing_windows.ps1` for a one-time Windows-only bundle.
It downloads the pinned runtime wheel, builds without executing the application
or tests, and produces `dist/BeamProfiler-SpotTracing-Windows-x64-Setup.exe`.
The executable enables live spot tracing by default. It has a separate product
name and installation directory, includes no custom demo, and leaves camera
drivers untouched.

The existing GitHub workflow has a manual `windows_spot_tracing` input. Dispatch
`build.yml` at `live-spot-diameters` with that input set to `true` for the same
Windows-only artifact. No macOS build, verification jobs or release runs in that mode.
