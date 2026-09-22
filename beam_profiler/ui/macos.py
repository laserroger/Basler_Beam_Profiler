"""Native close-button support for OpenCV's Cocoa viewer window."""
from AppKit import NSApplication, NSWindowCloseButton, NSWindowStyleMaskClosable
from Foundation import NSObject
import objc


class ProfilerCloseHandler(NSObject):
    @objc.python_method
    def configure(self, callback):
        self.callback = callback

    def closeProfiler_(self, sender):
        # Let the application loop release the camera and destroy its windows.
        # Directly closing Cocoa's window would bypass OpenCV's bookkeeping.
        self.callback()


def attach_close_button(title, callback):
    for window in NSApplication.sharedApplication().windows():
        if str(window.title()) == title:
            window.setStyleMask_(window.styleMask() | NSWindowStyleMaskClosable)
            button = window.standardWindowButton_(NSWindowCloseButton)
            handler = ProfilerCloseHandler.alloc().init()
            handler.configure(callback)
            button.setTarget_(handler)
            button.setAction_('closeProfiler:')
            button.setEnabled_(True)
            return handler  # Cocoa target is weak; retain for the viewer lifetime.
    raise RuntimeError('Cannot locate the beam-profiler Cocoa window')


def attach_key_monitor(title, callback):
    """Keep viewer keys dispatched by Tk instead of OpenCV's waitKey.

    Cocoa waitKey returns key events directly without dispatching them. Tk can
    also pump the same event queue between frames, so capture keys it dispatches
    to this viewer and let the main loop process them exactly once.
    """
    from AppKit import NSEvent, NSEventMaskKeyDown

    def handler(event):
        window = event.window()
        if window is not None and str(window.title()) == title:
            chars = event.characters()
            if chars:
                callback(ord(str(chars)[0]))
                return None
        return event

    token = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, handler)
    return token


def remove_key_monitor(token):
    from AppKit import NSEvent
    NSEvent.removeMonitor_(token)
