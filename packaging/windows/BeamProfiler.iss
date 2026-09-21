; Compile with tools/build_windows.ps1 after the app passes its driver checks.
#ifndef VendorInstaller
  #error VendorInstaller must identify the Windows x64 full Spinnaker installer
#endif
#ifndef AppSource
  #error AppSource must identify the built application directory
#endif
#ifndef OutputPath
  #error OutputPath must identify the output directory
#endif

[Setup]
AppId={{0D65F53A-B30C-43FA-923D-EFF972710748}
AppName=Beam Profiler
AppVersion=1.0
DefaultDirName={autopf}\Beam Profiler
DefaultGroupName=Beam Profiler
OutputDir={#OutputPath}
OutputBaseFilename=BeamProfiler-Windows-x64-Setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\pylon_camera.exe

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#VendorInstaller}"; DestDir: "{tmp}"; DestName: "SpinnakerSDK.exe"; Flags: deleteafterinstall

[Icons]
Name: "{group}\Beam Profiler"; Filename: "{app}\pylon_camera.exe"
Name: "{group}\Beam Profiler (FLIR)"; Filename: "{app}\pylon_camera.exe"; Parameters: "--camera flir"
Name: "{group}\Beam Profiler (Simulator)"; Filename: "{app}\pylon_camera.exe"; Parameters: "--sim"

[Run]
Filename: "{app}\pylon_camera.exe"; Description: "Open Beam Profiler"; Flags: nowait postinstall skipifsilent runasoriginaluser

[Code]
var
  DriverRestartNeeded: Boolean;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Installing FLIR camera runtime and USB drivers...';
    if not Exec(ExpandConstant('{tmp}\SpinnakerSDK.exe'),
      '/install /silent Features=Runtimes_v140,USBDrivers', '',
      SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      RaiseException('Could not start the bundled FLIR driver installer.');
    if (ResultCode <> 0) and (ResultCode <> 3010) then
      RaiseException(Format('FLIR driver installation failed (code %d). See C:\ProgramData\Spinnaker\Logs.', [ResultCode]));
    DriverRestartNeeded := ResultCode = 3010;
  end;
end;

function NeedRestart(): Boolean;
begin
  Result := DriverRestartNeeded;
end;
