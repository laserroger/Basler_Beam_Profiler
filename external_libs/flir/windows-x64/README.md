# Windows FLIR build inputs

The pinned vendor download in `manifest.json` is the Spinnaker 4.4.0.246
CPython 3.12 Windows x64 wheel, originally downloaded from Teledyne. The binary
is stored as a release asset in the fork; its SHA-256 is verified before use.
The accompanying notices are copied from the same vendor Python ZIP.

The wheel includes matching native FLIR DLLs and the GenTL transport, collected
into the app. The build does not download or bundle the full Spinnaker SDK
installer. The application uses the destination PC's existing camera drivers.
