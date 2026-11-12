"""Windows spot-tracing product: open the camera and live trace windows."""
import sys
from beam_profiler.__main__ import main

if __name__ == '__main__':
    main(['--spot-sizes', *sys.argv[1:]])
