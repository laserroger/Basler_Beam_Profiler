#!/bin/bash
# Launch the beam profiler using the "beamprofiler" conda env set up for this machine.
cd "$(dirname "$0")"
exec /home/simonlab/miniconda3/envs/beamprofiler/bin/python pylon_camera.py "$@"
