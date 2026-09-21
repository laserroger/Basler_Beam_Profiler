"""Windows x64 application with the vendor's complete PySpin runtime."""
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

if sys.platform != 'win32':
    raise RuntimeError('Build the Windows application on Windows x64')
data, binaries, hidden = collect_all('PySpin')
if not any(source.endswith('Spinnaker_GenTL_v140.cti') for source, _ in data):
    raise RuntimeError('The matching FLIR GenTL transport was not collected')
a = Analysis(
    ['pylon_camera.py'], pathex=[], binaries=binaries,
    datas=data + copy_metadata('spinnaker_python') + [
        ('camera_config.yaml', '.'), ('docs/windows.md', 'docs'),
        ('.vendor/windows/licenses', 'licenses/spinnaker')],
    hiddenimports=hidden, hookspath=[],
    runtime_hooks=['packaging_windows_hook.py'], excludes=['cupy'], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='pylon_camera',
          debug=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='pylon_camera')
