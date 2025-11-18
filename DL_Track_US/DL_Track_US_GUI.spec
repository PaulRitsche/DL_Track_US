# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['DL_Track_US_GUI.py'],
    pathex=[],
    binaries=[],
    datas=[('gui_helpers/gui_files/settings.json', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/DLTrack_logo.png', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/gui_color_theme.json', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/DLTrack_logo.ico', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/gear.png', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/Cite.png', 'gui_helpers/gui_files'), ('gui_helpers/gui_files/Info.png', 'gui_helpers/gui_files')],
    hiddenimports=['customtkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DL_Track_US_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gui_helpers\\gui_files\\DLTrack_logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DL_Track_US_GUI',
)
