# Windows FLIR build inputs

The pinned vendor downloads in `manifest.json` are Spinnaker 4.4.0.246 Windows
x64 and its CPython 3.12 wheel, downloaded from Teledyne. Large binaries are
stored as release assets in the fork; SHA-256 checksums are verified before use.
The accompanying notices are copied from the same vendor Python ZIP.

The wheel includes the matching native FLIR DLLs and GenTL transport. These are
collected into the app. The full signed vendor installer is embedded inside our
offline setup and installs the required runtime prerequisites and USB drivers.
The app does not need a separate Python or SDK download on the destination PC.
