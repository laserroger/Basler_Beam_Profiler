# -*- mode: python ; coding: utf-8 -*-

binaries = [
    ('external_libs/libMathParser_gcc_v3_1_Basler_pylon_v3.dylib', 'pypylon'),
    ('external_libs/libNodeMapData_gcc_v3_1_Basler_pylon_v3.dylib', 'pypylon'),
    ('external_libs/libGenApi_gcc_v3_1_Basler_pylon_v3.dylib', 'pypylon'),
    ('external_libs/libXmlParser_gcc_v3_1_Basler_pylon_v3.dylib', 'pypylon'),
]
a = Analysis(
    ['pylon_camera.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=[],
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
    name='pylon_camera',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pylon_camera',
)
app = BUNDLE(
    coll,
    name='pylon_camera.app',
    icon=None,
    bundle_identifier=None,
)
