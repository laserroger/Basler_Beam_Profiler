from pathlib import Path
from beam_profiler import config


def test_mac_bundle_uses_writable_config_without_overwriting(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'camera_config.yaml').write_text('cameras: {}\n')
    writable = tmp_path / 'support'
    monkeypatch.setattr(config.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(config.sys, 'platform', 'darwin')
    monkeypatch.setattr(config.sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setattr(config.os.path, 'expanduser', lambda _: str(writable))
    assert config.app_dir() == str(writable)
    path = writable / 'camera_config.yaml'
    assert path.read_text() == 'cameras: {}\n'
    path.write_text('cameras: {custom: 1}\n')
    config.app_dir()
    assert path.read_text() == 'cameras: {custom: 1}\n'


def test_windows_installer_uses_writable_user_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(config.sys, 'platform', 'win32')
    monkeypatch.setattr(config.sys, 'executable', str(tmp_path / 'profiler.exe'))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'camera_config.yaml').write_text('cameras: {}\n')
    monkeypatch.setattr(config.sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    path = Path(config.app_dir())
    assert path == tmp_path / 'local' / 'BeamProfiler'
    assert (path / 'camera_config.yaml').read_text() == 'cameras: {}\n'
    (path / 'camera_config.yaml').write_text('cameras: {custom: 1}\n')
    config.app_dir()
    assert (path / 'camera_config.yaml').read_text() == 'cameras: {custom: 1}\n'


def test_upgrade_adds_bundled_models_and_preserves_user_calibration(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'camera_config.yaml').write_text('''cameras:
  existing:
    default_roi: [640, 480, 0, 0]
    pixel_size: 3.45e-6
  new_model:
    default_roi: [4096, 3000, 0, 0]
    pixel_size: 3.45e-6
''')
    user = tmp_path / 'user'
    user.mkdir()
    saved = '''cameras:
  existing:
    default_roi: [320, 240, 8, 4]
    pixel_size: 4.0e-6
'''
    (user / 'camera_config.yaml').write_text(saved)
    monkeypatch.setattr(config.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(config.sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setattr(config, 'APP_DIR', str(user))
    models = config.load_camera_config()
    assert models['new_model']['default_roi'] == (4096, 3000, 0, 0)
    assert models['existing']['pixel_size'] == 4e-6
    assert models['existing']['default_roi'] == (320, 240, 8, 4)
    assert (user / 'camera_config.yaml').read_text() == saved
