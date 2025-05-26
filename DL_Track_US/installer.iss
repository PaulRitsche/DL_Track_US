[Setup]
AppName=DL_Track_US
AppVersion=0.3.0
DefaultDirName={pf}\DL_Track_US
DefaultGroupName=DL_Track_US
OutputDir=installer
OutputBaseFilename=DLTrackUS_Installer
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\DL_Track_US_GUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DL_Track_US"; Filename: "{app}\DL_Track_US_GUI.exe"
Name: "{commondesktop}\DL_Track_US"; Filename: "{app}\DL_Track_US_GUI.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
