"""Render README GIFs through the simulator and viewer, without opening a camera.

Run: python -m tools.make_readme_demos (requires ffmpeg on PATH).
GIF playback is illustrative, not a live-camera FPS benchmark.
"""
from pathlib import Path
import math
import re
import subprocess
import tempfile

import cv2
import numpy as np

from beam_profiler import fitconfig
from beam_profiler.cameras.simulated import SimulatedCamera
from beam_profiler.processing.pipeline import FrameProcessor
from beam_profiler.synthetic import Spot
from beam_profiler.ui.viewer import Viewer

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'demos'
FRAMES, FPS = 72, 12


def make_viewer(kind):
    camera = SimulatedCamera(width=1200, height=900, jitter=0, noise=0, seed=42)
    camera.model_name = {'beam':'Simulated-single-beam', 'array':'Simulated-8x1',
                         'grid':'Simulated-6x3', 'roi':'Simulated-8x1',
                         'saturation':'Simulated-saturation'}[kind]
    v = Viewer.__new__(Viewer)  # Rendering only: no native windows or camera SDK.
    v.camera = camera
    v._auto_exp = False
    v.profiler_enabled = kind in ('beam', 'saturation')
    v.do_fitting = not v.profiler_enabled
    v.show_stats = kind in ('array', 'grid')
    v.row_col_fitting = kind == 'grid'
    v.show_rect_stats = v.web_server_enabled = False
    v.rect_disp = v.rect_sensor = v.fit_rect_disp = v.fit_rect_sensor = None
    v._mouse_display = v.current_frame = None
    v.profiler_message = ''
    v.avg_dt = 1/60
    v.processor = FrameProcessor()
    # Offline rendering cannot truthfully report a live acquisition frame rate.
    original_hud = v._hud_lines
    def hud():
        lines = original_hud()
        lines[0] = re.sub(r', FPS:.*', ' - simulated data / offline replay', lines[0])
        return lines
    v._hud_lines = hud
    return v


def scene(kind, i):
    phase = 2 * math.pi * i / FRAMES
    # Smooth, periodic perturbation: distorted -> aligned -> distorted.
    error = (1 + math.cos(phase)) / 2
    if kind == 'saturation':
        return [Spot(600, 570, 65, 48, .25, .65 + error)]
    if kind == 'beam':
        return [Spot(600 + 130*math.sin(phase), 570 + 55*math.cos(phase),
                     48 + 6*math.sin(phase), 33, phase/3, .7)]
    if kind in ('array', 'roi'):
        return [Spot(180 + col*120 + (8*error*math.sin(col*1.3) if kind=='array' else 0),
                     650 + (9*error*math.cos(col*.9) if kind=='array' else 0),
                     9, 8, amplitude=.65) for col in range(8)]
    return [Spot(200 + col*155 + error*8*math.sin(row+col*.7),
                 430 + row*155 + error*12*math.cos(col*.8+row),
                 9, 8, amplitude=.65) for row in range(3) for col in range(6)]


def render_demo(kind, tmp):
    titles = {'beam':'Single beam: position and per-frame Gaussian fit',
              'array':'Beam array: spot positions and spacing statistics',
              'grid':'Grid alignment: row/column grouping and deviations',
              'roi':'Hardware ROI: fewer captured pixels to process',
              'saturation':'Saturation: green fit / thick red warning'}
    v = make_viewer(kind)
    report = []
    for i in range(FRAMES):
        v.camera.spots = scene(kind, i)
        if kind == 'roi':
            # Hold each crop long enough to read it, then return for the loop.
            stage = min(i//18, 3)
            v.camera.ROI = [(1200,900,0,0),(1100,450,50,425),
                            (1000,180,100,560),(1200,900,0,0)][stage]
        frame = v.camera.grab_image()
        v.current_frame = frame
        v._analyze_frame(frame)
        spots = v.analysis.spots
        if len(spots) != len(v.camera.spots):
            raise RuntimeError(f'{kind} frame {i}: expected {len(v.camera.spots)} spots, got {len(spots)}')
        display = v._compose(cv2.convertScaleAbs(frame, alpha=1/256.), spots)
        canvas = np.full((1220,1100,3), (26,22,18), np.uint8)
        cv2.putText(canvas, titles[kind], (22,38), cv2.FONT_HERSHEY_SIMPLEX, .8, (245,245,245), 2, cv2.LINE_AA)
        if kind == 'saturation':
            detail = f'Raw peak: {frame.max()} / {v.camera.saturation}   Reduce exposure until the outline is green'
        elif kind == 'beam':
            detail = f'P: profiler   Center: ({spots.x[0]:.1f}, {spots.y[0]:.1f}) sensor px'
        elif kind == 'roi':
            w,h,_,_ = v.camera.ROI
            detail = f'Ctrl/Shift + scroll: ROI {w} x {h}   ({w*h/1e6:.2f} MP; {100*w*h/(1200*900):.0f}% of full frame)'
        else:
            detail = ('F + G + H' if kind=='grid' else 'F + G') + ': ' + ('Near aligned' if 24 <= i < 48 else 'Introduced position / spacing deviations')
        cv2.putText(canvas, detail, (22,77), cv2.FONT_HERSHEY_SIMPLEX, .65, (130,225,240), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'SIMULATOR  |  12 fps GIF playback, not a camera-speed benchmark', (22,106), cv2.FONT_HERSHEY_SIMPLEX, .5, (180,180,180), 1, cv2.LINE_AA)
        canvas[120:] = display
        image = cv2.resize(canvas, (880,976), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(tmp / f'{i:03}.png'), image)
        if i in (0,36):
            cv2.imwrite(str(tmp.parent / f'{kind}-{i}.png'), image)
        report.append({'frame':i,'spots':len(spots),'roi':v.camera.ROI,
                       'std_dx_px':float(np.std(v.statistics.stats_dx)), 'std_dy_px':float(np.std(v.statistics.stats_dy))})
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(FPS),
                    '-i',str(tmp/'%03d.png'),'-filter_complex',
                    '[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                    '-loop','0',str(OUT/f'{kind}.gif')],check=True)
    if kind == 'saturation':
        panels = []
        for index, label in [(36, 'Below saturation: green outline'),
                             (0, 'Clipped peak: thick red outline')]:
            im = cv2.imread(str(tmp.parent / f'saturation-{index}.png'))
            panel = np.full((410, 440, 3), 22, np.uint8)
            cv2.putText(panel, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .58, (255, 255, 255), 1, cv2.LINE_AA)
            panel[45:405] = im[450:810, 220:660]
            panels.append(panel)
        cv2.imwrite(str(OUT / 'saturation-comparison.png'), np.hstack(panels))
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old = fitconfig.active()
    try:
        fitconfig.set_active(fitconfig.FitConfig(heatmap=True,gpu_enabled=False),save=False)
        with tempfile.TemporaryDirectory(prefix='profiler-demos-') as work:
            root=Path(work)
            for kind in ('beam','array','roi','saturation'):
                tmp=root/kind
                tmp.mkdir()
                report=render_demo(kind,tmp)
                # Inspection images stay outside the repository.
                for i in (0,36):
                    Path(f'/tmp/profiler-demo-{kind}-{i}.png').write_bytes((root/f'{kind}-{i}.png').read_bytes())
                print(kind, 'frames:',len(report),'bytes:',(OUT/f'{kind}.gif').stat().st_size,flush=True)
                if kind=='grid': print('Grid spacing deviations (start, midpoint):', report[0],report[36],flush=True)
    finally:
        fitconfig.set_active(old,save=False)


if __name__ == '__main__':
    main()
