# Build a native macOS app with the current Python environment's architecture.
# pypylon's PyInstaller hook collects its matching SDK libraries; do not bundle
# the old, manually copied external_libs binaries from another SDK version.
a = Analysis(
    ['pylon_camera.py'],
    pathex=[],
    binaries=[],
    datas=[('camera_config.yaml', '.'), ('docs/macos.md', 'docs')],
    hiddenimports=['AppKit', 'Foundation'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['PySpin', '_PySpin', 'cupy'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='pylon_camera',
    debug=False, strip=False, upx=False, console=False,
    argv_emulation=False, target_arch=None, codesign_identity=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='pylon_camera')
app = BUNDLE(
    coll, name='pylon_camera.app', icon=None,
    bundle_identifier='com.laserroger.beamprofiler',
    info_plist={'NSHighResolutionCapable': True},
)
