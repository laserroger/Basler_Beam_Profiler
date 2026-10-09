"""Use real subprocess exits: sys.exit would hang in these fake destructors."""
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('failure', ['acquisition','cleanup'])
def test_fatal_shutdown_bypasses_hanging_destructors(failure, tmp_path):
    marker=tmp_path/'destructor-ran'
    script = '''
import atexit
import time
from pathlib import Path
from beam_profiler.__main__ import close_cameras
marker = Path(MARKER)
class Camera:
    def close(self):
        if FAILURE == 'cleanup':
            raise RuntimeError('Spinnaker: Camera is disconnected [-1014]')
        marker.write_text('close should not run')
        time.sleep(60)
    def __del__(self):
        marker.write_text('destructor should not run')
        time.sleep(60)
atexit.register(lambda: marker.write_text('Python finalization should not run'))
close_cameras([Camera()], failed=FAILURE == 'acquisition')
'''.replace('MARKER',repr(str(marker))).replace('FAILURE',repr(failure))
    completed = subprocess.run([sys.executable,'-c',script], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True,text=True,timeout=15)
    assert completed.returncode == 1
    assert 'terminating the camera process' in completed.stderr
    assert not marker.exists()


def test_normal_exit_closes_cameras_without_forced_exit(monkeypatch):
    from beam_profiler import __main__ as app
    calls=[]
    class Camera:
        def close(self): calls.append('closed')
    monkeypatch.setattr(app,'exit_failed_camera_process',lambda: pytest.fail('normal shutdown forced exit'))
    app.close_cameras([Camera(),Camera()])
    assert calls == ['closed','closed']
