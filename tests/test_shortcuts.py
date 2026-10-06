"""Uppercase input dispatches exactly like lowercase without opening windows."""
import pytest
from types import SimpleNamespace
from beam_profiler.ui import viewer


@pytest.mark.parametrize('letter', list('afpghcvsdtwoy'))
def test_letter_shortcuts_are_case_insensitive(letter, monkeypatch):
    monkeypatch.setattr(viewer, 'write_status', lambda _: None)
    events = []
    monkeypatch.setattr(viewer.settings, 'open_settings', lambda: events.append('settings'))
    def run(key):
        v = viewer.Viewer.__new__(viewer.Viewer)
        v.profiler_enabled = v.do_fitting = v.show_stats = v.row_col_fitting = False
        v._auto_exp = False
        v.exposure_us = 200
        v.rect_sensor = v.fit_rect_sensor = (0, 0, 10, 10)
        v.camera = SimpleNamespace(AutoExposureOn=lambda **kw: events.append('auto'),
                                  userdefined_line_enabled=False,
                                  set_userdefined_line=lambda _: events.append('output'))
        for method in ('_toggle_angular', '_quick_save', '_dialog_save', '_switch_camera', '_toggle_web_server'):
            setattr(v, method, lambda name=method: events.append(name))
        assert v._handle_key(ord(key))
        return (v.profiler_enabled, v.do_fitting, v.show_stats, v.row_col_fitting,
                v.auto_exp, v.rect_sensor, v.fit_rect_sensor)
    lower = run(letter)
    lower_events = events.copy()
    events.clear()
    upper = run(letter.upper())
    assert upper == lower
    assert events == lower_events
