"""Load FLIR from the packaged PySpin directory, without SDK installation."""
import os
import sys
from pathlib import Path

runtime = Path(sys._MEIPASS) / 'PySpin'
# Keep the handle alive: closing it removes the directory from DLL search.
sys._beam_flir_dll_handle = os.add_dll_directory(str(runtime))
transport = runtime / 'Spinnaker_GenTL_v140.cti'
if not transport.is_file():
    raise RuntimeError('The bundled Windows FLIR transport is missing')
os.environ['SPINNAKER_GENTL64_CTI'] = str(transport)
