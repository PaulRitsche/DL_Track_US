[Setup]
AppName=DLTrackUS0.3.1
AppVersion=0.3.1
DefaultDirName={pf}\DLTrackUS
DefaultGroupName=DLTrackUS
OutputDir=installer
OutputBaseFilename=DLTrackUS0.3.1_Installer
Compression=lzma
SolidCompression=yes
SetupIconFile=gui_helpers\gui_files\DLTrack_logo.ico

[Files]
Source: "dist\DL_Track_US_0.3.1\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DLTrackUS"; Filename: "{app}\DLTrackUS.exe"
Name: "{commondesktop}\DLTrackUS"; Filename: "{app}\DLTrackUS.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
