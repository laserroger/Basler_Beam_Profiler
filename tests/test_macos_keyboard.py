"""Native regression: Tk must not swallow the viewer's arrow keys."""
import sys
import pytest


@pytest.mark.skipif(sys.platform != 'darwin', reason='Cocoa event routing')
def test_up_key_survives_tk_event_pump_once_per_press():
    import cv2
    import numpy as np
    import tkinter as tk
    from AppKit import NSApplication, NSEvent, NSEventTypeKeyDown
    from beam_profiler.ui import settings
    from beam_profiler.ui.macos import attach_key_monitor, remove_key_monitor

    try:
        settings.prepare_gui()
    except tk.TclError:
        pytest.skip('GUI session unavailable')
    title = 'Profiler keyboard regression'
    token = None
    try:
        cv2.namedWindow(title)
        cv2.imshow(title, np.zeros((80, 80), np.uint8))
        cv2.waitKeyEx(1)
        app = NSApplication.sharedApplication()
        window = next(w for w in app.windows() if str(w.title()) == title)
        keys = []
        token = attach_key_monitor(title, keys.append)
        for _ in range(3):
            event = NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                NSEventTypeKeyDown, (0, 0), 0, 0, window.windowNumber(),
                None, '\uf700', '\uf700', False, 126)
            app.postEvent_atStart_(event, False)
            settings.pump_events()
            key = cv2.waitKeyEx(1)
            if key != -1:
                keys.append(key)
        assert keys == [63232, 63232, 63232]
    finally:
        if token is not None:
            remove_key_monitor(token)
        cv2.destroyWindow(title)
        settings.destroy_settings()
