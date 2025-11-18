[Setup]
AppName=DL_Track_US
AppVersion=0.3.1
DefaultDirName={pf}\DL_Track_US
DefaultGroupName=DL_Track_US
OutputDir=installer
OutputBaseFilename=DLTrackUS_Installer
Compression=lzma
SolidCompression=yes
SetupIconFile=gui_helpers\gui_files\DLTrack_logo.ico

[Files]
Source: "dist\DL_Track_US_GUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DLTrackUS_0.3.1"; Filename: "{app}\DL_Track_US_GUI.exe"
Name: "{commondesktop}\DLTrackUS_0.3.1"; Filename: "{app}\DL_Track_US_GUI.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
