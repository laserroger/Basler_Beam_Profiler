"""Entry point kept for run.sh and PyInstaller (see `compile`).

The application lives in the beam_profiler package; prefer
``python -m beam_profiler`` (add ``--sim`` for the simulated camera)."""

from beam_profiler.__main__ import main

if __name__ == "__main__":
    main()
