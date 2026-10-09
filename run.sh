#!/bin/bash
set -e
cd "$(dirname "$0")"
if [[ -d .vendor/spinnaker/lib ]]; then
    export SPINNAKER_GENTL64_CTI="$PWD/.vendor/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti"
    export DYLD_LIBRARY_PATH="$PWD/.vendor/spinnaker/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
if [[ -n "${PYTHON:-}" ]]; then
    exec "$PYTHON" -m beam_profiler "$@"
fi
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    exec "$VIRTUAL_ENV/bin/python" -m beam_profiler "$@"
fi
if [[ -x "$HOME/envs/distortion/bin/python" ]]; then
    exec "$HOME/envs/distortion/bin/python" -m beam_profiler "$@"
fi
exec python3 -m beam_profiler "$@"
