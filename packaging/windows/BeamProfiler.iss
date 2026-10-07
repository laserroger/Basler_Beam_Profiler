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
; Re-evaluate driver defaults on every installation, including app upgrades.
UsePreviousTasks=no
UninstallDisplayIcon={app}\pylon_camera.exe

[Tasks]
Name: "flirdrivers"; Description: "Install bundled FLIR USB drivers (existing camera software may be modified)"; Flags: checkedonce; Check: NoExistingCameraSoftware
Name: "flirdrivers"; Description: "Install bundled FLIR USB drivers (existing camera software may be modified)"; Flags: unchecked; Check: ExistingCameraSoftware

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#VendorInstaller}"; DestDir: "{tmp}"; DestName: "SpinnakerSDK.exe"; Flags: deleteafterinstall; Tasks: flirdrivers

[Icons]
Name: "{group}\Beam Profiler"; Filename: "{app}\pylon_camera.exe"
Name: "{group}\Beam Profiler (FLIR)"; Filename: "{app}\pylon_camera.exe"; Parameters: "--camera flir"
Name: "{group}\Beam Profiler (Simulator)"; Filename: "{app}\pylon_camera.exe"; Parameters: "--sim"

[Run]
Filename: "{app}\pylon_camera.exe"; Description: "Open Beam Profiler"; Flags: nowait postinstall skipifsilent runasoriginaluser

[Code]
var
  DriverRestartNeeded: Boolean;

function CameraSoftwareInRegistry(RootKey: Integer): Boolean;
var
  Names: TArrayOfString;
  I: Integer;
  DisplayName, Name: String;
begin
  Result := False;
  if not RegGetSubkeyNames(RootKey,
    'Software\Microsoft\Windows\CurrentVersion\Uninstall', Names) then
    Exit;
  for I := 0 to GetArrayLength(Names) - 1 do
  begin
    if RegQueryStringValue(RootKey,
      'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + Names[I],
      'DisplayName', DisplayName) then
    begin
      Name := Lowercase(DisplayName);
      if (Pos('spinnaker', Name) > 0) or (Pos('pylon', Name) > 0) or
        (Pos('basler', Name) > 0) or
        ((Pos('flir', Name) > 0) and (Pos('usb', Name) > 0)) then
      begin
        Log('Preserving existing camera software: ' + DisplayName);
        Result := True;
        Exit;
      end;
    end;
  end;
end;

function ExistingCameraSoftware(): Boolean;
begin
  { Check both registry views, including per-user installations. Never execute
    a vendor maintenance installer automatically on an existing camera setup. }
  Result := CameraSoftwareInRegistry(HKLM64) or
    CameraSoftwareInRegistry(HKLM32) or
    CameraSoftwareInRegistry(HKCU64) or
    CameraSoftwareInRegistry(HKCU32) or
    DirExists(ExpandConstant('{commonpf64}\Teledyne\Spinnaker')) or
    DirExists(ExpandConstant('{commonpf64}\FLIR Systems\Spinnaker')) or
    DirExists(ExpandConstant('{commonpf64}\Point Grey Research\Spinnaker')) or
    DirExists(ExpandConstant('{commonpf64}\Basler'));
end;

function NoExistingCameraSoftware(): Boolean;
begin
  Result := not ExistingCameraSoftware();
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardIsTaskSelected('flirdrivers') then
    begin
      Log('Bundled FLIR driver setup skipped; existing camera software preserved.');
      Exit;
    end;
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
