"""Average distinct raw snapshots from the running viewer without taking the camera."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np

from .calibration import FrameAverage


def capture(url, count, expected=None, *, timeout=180):
    average = FrameAverage()
    records = []
    seen = set()
    deadline = time.monotonic() + timeout
    saturated = None
    while average.n < count:
        if time.monotonic() > deadline:
            raise TimeoutError(f'Only {average.n}/{count} distinct raw frames arrived')
        with urlopen(url.rstrip('/') + '/api/raw_frame', timeout=30) as response:
            stamp = response.headers['X-Frame-Timestamp']
            if not stamp:
                raise ValueError('Viewer did not provide a frame timestamp')
            if stamp in seen:
                time.sleep(.03)
                continue
            settings = tuple(response.headers.get(name, '') for name in
                             ['X-Camera-Serial', 'X-Exposure-Us', 'X-Gain-Db', 'X-Pixel-Format', 'X-ADC-Bit-Depth'])
            if expected is None:
                expected = settings
            if settings != expected:
                raise ValueError('Camera or acquisition settings changed during capture')
            frame = np.load(BytesIO(response.read()), allow_pickle=False)
        if frame.ndim != 2 or frame.dtype not in (np.uint8, np.uint16):
            raise ValueError('API did not return an uncorrected integer sensor frame')
        seen.add(stamp)
        average.add(frame)
        if saturated is None:
            saturated = np.zeros(frame.shape, bool)
        saturated |= frame >= .98 * np.iinfo(frame.dtype).max
        records.append(dict(timestamp=float(stamp), mean_raw_dn=float(frame.mean())))
        if average.n % 10 == 0 or average.n == count:
            print(f'Captured {average.n}/{count} distinct raw frames', flush=True)
    return average, records, expected, saturated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5000')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--validation-frames', type=int, default=10)
    parser.add_argument('--dark', type=Path, help='dark mean .npy in matching raw units and geometry')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.validation_frames < 1:
        parser.error('Need at least two calibration frames and one validation frame')
    with urlopen(args.url.rstrip('/') + '/api/status', timeout=10) as response:
        before = json.load(response)
    if before.get('auto_exposure'):
        parser.error('Disable automatic exposure before calibration capture')
    args.output.mkdir(parents=True, exist_ok=False)
    average, records, expected, saturated = capture(args.url, args.frames)
    np.save(args.output / 'mean_raw.npy', average.mean.astype(np.float32))
    np.save(args.output / 'variance_raw.npy', average.variance.astype(np.float32))
    average.m2 = None
    # Never reuse a training frame in the independent validation average.
    last = records[-1]['timestamp']
    validation = FrameAverage()
    val_records = []
    while validation.n < args.validation_frames:
        new, rows, _, _ = capture(args.url, 1, expected)
        if rows[0]['timestamp'] <= last:
            time.sleep(.03)
            continue
        validation.add(new.mean)
        val_records.extend(rows)
        last = rows[0]['timestamp']
    with urlopen(args.url.rstrip('/') + '/api/status', timeout=10) as response:
        after = json.load(response)
    for key in ['camera_serial', 'exposure_time', 'gain_db', 'pixel_format', 'adc_bit_depth', 'roi', 'auto_exposure']:
        if before.get(key) != after.get(key):
            raise ValueError('Acquisition settings changed; no gain map generated')
    dark = (np.load(args.dark, allow_pickle=False) if args.dark else np.zeros_like(average.mean))
    if dark.shape != average.mean.shape or not np.isfinite(dark).all():
        raise ValueError('Dark mean has different geometry or nonfinite values')
    if tuple(before['roi'][2:]) != (0, 0):
        raise ValueError('Capture a full-sensor map with zero camera ROI offsets')
    signal = average.mean - dark
    normalization = float(np.median(signal[signal > 0]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError('No usable illuminated signal')
    valid = np.isfinite(signal) & (signal > .01 * normalization) & ~saturated
    gain = np.full(signal.shape, np.nan, np.float32)
    gain[valid] = signal[valid] / normalization
    corrected = np.full(signal.shape, np.nan, np.float32)
    corrected[valid] = (validation.mean[valid] - dark[valid]) / gain[valid]
    for name, array in [('gain_map', gain), ('dark_mean', dark), ('invalid_pixel_mask', ~valid),
                        ('validation_mean_raw', validation.mean), ('validation_corrected', corrected)]:
        np.save(args.output / (name + '.npy'), np.asarray(array, dtype=np.float32) if name != 'invalid_pixel_mask' else array)
    h, w = signal.shape
    sl = (slice(h//10, h-h//10), slice(w//10, w-w//10))
    def rms(image):
        x = image[sl]
        return float(np.nanstd(x / np.nanmedian(x)) * 100)
    summary = dict(camera_serial=before['camera_serial'], exposure_us=before['exposure_time'],
                   camera_before={'Gain': before['gain_db'], 'PixelFormat': before['pixel_format'],
                                  'AdcBitDepth': before.get('adc_bit_depth')},
                   dark_map='dark_mean.npy', calibration_frames=args.frames,
                   independent_validation_frames=args.validation_frames, dark_source=str(args.dark) if args.dark else None,
                   normalization_raw_dn=normalization, invalid_pixels=int((~valid).sum()),
                   validation_rms_before_pct=rms(validation.mean-dark), validation_rms_after_pct=rms(corrected),
                   note='Empirical flat field includes illumination; no sensor-only separation is asserted.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'capture.json').write_text(json.dumps(dict(before=before, after=after,
                         calibration=records, validation=val_records), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
