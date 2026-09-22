from beam_profiler.ui.viewer import Viewer


class QuantizedExposure:
    min_exposure, max_exposure = 11., 100000.

    def __init__(self, exposure=34.):
        self.value = exposure
        self.requests = []

    @property
    def ExposureTime(self):
        return self.value

    @ExposureTime.setter
    def ExposureTime(self, value):
        self.requests.append(value)
        self.value = 11. + 23. * max(0, round((value - 11.) / 23.))


def viewer(value=34.):
    v = Viewer.__new__(Viewer)
    v._auto_exp = False
    v.camera = QuantizedExposure(value)
    return v


def test_small_changes_escape_hardware_rounding_in_both_directions():
    for factor in [.9, 1.1]:
        v = viewer()
        for _ in range(12):
            v._adjust_exposure(factor)
        assert (v.camera.ExposureTime-34)*(factor-1) > 0


def test_reversal_starts_from_displayed_value():
    v = viewer()
    for _ in range(3):
        v._adjust_exposure(1.1)
    before = v.camera.ExposureTime
    v._adjust_exposure(.9)
    assert v.camera.requests[-1] == before*.9
    assert v.camera.ExposureTime <= before


def test_minimum_does_not_accumulate_hidden_undershoot():
    v = viewer(11.)
    for _ in range(30):
        v._adjust_exposure(.1)
    assert min(v.camera.requests) == 11.
    v._adjust_exposure(10)
    assert v.camera.requests[-1] == 110.


def test_external_exposure_change_resets_target():
    v = viewer()
    v._adjust_exposure(1.1)
    v.camera.value = 1000.
    v._adjust_exposure(1.1)
    assert v.camera.requests[-1] == 1100.


def test_manual_keys_do_not_fight_auto_exposure():
    v = viewer()
    v._auto_exp = True
    v._adjust_exposure(.1)
    assert not v.camera.requests


def test_keys_from_both_event_loops_are_processed_once_in_order(monkeypatch):
    from collections import deque
    v = viewer(11.)
    v._pending_keys = deque([63232])
    calls = []
    v._handle_key = lambda key: calls.append(key) or True
    monkeypatch.setattr('beam_profiler.ui.viewer.cv2.waitKeyEx', lambda delay: 63233)
    assert v._poll_keys(1)
    assert calls == [63232, 63233]
    monkeypatch.setattr('beam_profiler.ui.viewer.cv2.waitKeyEx', lambda delay: -1)
    assert v._poll_keys(1)
    assert calls == [63232, 63233]
