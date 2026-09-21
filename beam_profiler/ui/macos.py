"""Native close-button support for OpenCV's Cocoa viewer window."""
from AppKit import NSApplication, NSWindowCloseButton, NSWindowStyleMaskClosable
from Foundation import NSObject
import objc


def show_camera_error(message):
    """Make discovery failures visible when Finder hides console output."""
    from AppKit import NSAlert, NSApplicationActivationPolicyRegular
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    alert = NSAlert.alloc().init()
    alert.setMessageText_("Unable to open camera")
    alert.setInformativeText_(message)
    alert.addButtonWithTitle_("Quit")
    app.activateIgnoringOtherApps_(True)
    alert.runModal()


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
