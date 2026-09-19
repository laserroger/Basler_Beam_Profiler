# Build a native macOS app with the current Python environment's architecture.
# pypylon's PyInstaller hook collects its matching SDK libraries; do not bundle
# the old, manually copied external_libs binaries from another SDK version.
import os
from pathlib import Path

runtime = Path(os.environ.get('SPINNAKER_RUNTIME_DIR', '.vendor/spinnaker'))
with_flir = (runtime / 'lib' / 'libSpinnaker.4.dylib').is_file()
if os.environ.get('BEAM_REQUIRE_FLIR') == '1' and not with_flir:
    raise RuntimeError('The vendored FLIR runtime must be extracted before packaging')

a = Analysis(
    ['pylon_camera.py'],
    pathex=[],
    binaries=[],
    datas=[('camera_config.yaml', '.'), ('docs/macos.md', 'docs')],
    hiddenimports=['AppKit', 'Foundation'],
    hookspath=[],
    runtime_hooks=['packaging_flir_hook.py'] if with_flir else [],
    excludes=['cupy', 'PySpin', '_PySpin'],
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
    info_plist={'NSHighResolutionCapable': True, 'LSMinimumSystemVersion': '26.0'},
)

if with_flir:
    from tools.bundle_flir import bundle
    bundle(Path(DISTPATH) / 'pylon_camera.app', runtime)
