; Application-only Windows installer. Camera drivers remain installed separately.
#ifndef AppSource
  #error AppSource must identify the built application directory
#endif
#ifndef OutputPath
  #error OutputPath must identify the output directory
#endif

#ifdef SpotTracing
  #define ProductName "Beam Profiler Spot Tracing"
  #define ProductId "{{566AFD24-33DC-4958-B615-E932F499C841}"
  #define OutputName "BeamProfiler-SpotTracing-Windows-x64-Setup"
#else
  #define ProductName "Beam Profiler"
  #define ProductId "{{0D65F53A-B30C-43FA-923D-EFF972710748}"
  #define OutputName "BeamProfiler-Windows-x64-Setup"
#endif

[Setup]
AppId={#ProductId}
AppName={#ProductName}
AppVersion=1.0
DefaultDirName={autopf}\{#ProductName}
DefaultGroupName={#ProductName}
OutputDir={#OutputPath}
OutputBaseFilename={#OutputName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\pylon_camera.exe

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#ProductName}"; Filename: "{app}\pylon_camera.exe"
Name: "{group}\{#ProductName} (FLIR)"; Filename: "{app}\pylon_camera.exe"; Parameters: "--camera flir"
#ifndef SpotTracing
Name: "{group}\Beam Profiler (Simulator)"; Filename: "{app}\pylon_camera.exe"; Parameters: "--sim"
#endif

[Run]
Filename: "{app}\pylon_camera.exe"; Description: "Open {#ProductName}"; Flags: nowait postinstall skipifsilent runasoriginaluser
