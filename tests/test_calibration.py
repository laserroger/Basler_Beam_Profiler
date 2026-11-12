from io import BytesIO
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest

from beam_profiler.calibration import FlatField, FrameAverage, display_gray
from beam_profiler.processing.pipeline import AnalysisOptions, FrameProcessor
from beam_profiler.server import create_app
from beam_profiler.session import LiveSession, SessionSnapshot
from beam_profiler.synthetic import render, Spot


def test_full_sensor_gain_and_dark_follow_offset_roi_without_modifying_raw():
    gain = np.linspace(.5, 1.5, 80, dtype=np.float32).reshape(8, 10)
    dark = np.arange(80, dtype=np.float32).reshape(8, 10)
    calibration = FlatField(gain, dark)
    roi = (4, 3, 3, 2)
    signal = np.arange(12).reshape(3, 4) * 150 + 2000
    raw = np.rint(signal * gain[2:5, 3:7] + dark[2:5, 3:7]).astype(np.uint16)
    original = raw.copy()
    corrected = calibration.apply(raw, roi)
    np.testing.assert_allclose(corrected, signal, atol=1)
    np.testing.assert_array_equal(raw, original)
    assert corrected.dtype == np.float32
    with pytest.raises(ValueError, match='ROI'):
        calibration.apply(raw, (4, 3, 8, 2))


def test_raw_endpoint_remains_integer_when_display_and_statistics_are_corrected():
    raw = np.full((20, 40), 20000, np.uint16)
    calibration = FlatField(np.full(raw.shape, 2), np.full(raw.shape, 100))
    result = FrameProcessor(calibration).process(raw, roi=(40, 20, 0, 0), pixel_size=1,
            full_scale=65535, options=AnalysisOptions(rect=(0, 0, 40, 20), rect_stats=True))
    assert result.rect_stats['mean'] == 9950
    session = LiveSession()
    session.publish(SessionSnapshot(result, dict(camera_serial='test', exposure_time=29,
                             gain_db=0, pixel_format='Mono16', calibration=dict(enabled=True))))
    client = create_app(session).test_client()
    response = client.get('/api/raw_frame')
    np.testing.assert_array_equal(np.load(BytesIO(response.data)), raw)
    assert response.headers['X-Exposure-Us'] == '29'
    assert response.headers['X-Frame-Timestamp'] == str(result.timestamp)
    assert client.get('/api/calibration').json['enabled']
    image = cv2.imdecode(np.frombuffer(client.get('/api/image').data, np.uint8), 0)
    assert image.mean() == pytest.approx(39, abs=1)


@pytest.mark.parametrize('amplitude,gain,expected', [(.6, .25, False), (1.3, 2., True)])
def test_saturation_comes_from_raw_sensor_not_amplified_or_reduced_correction(amplitude, gain, expected):
    raw = render(240, 200, [Spot(120, 100, 12, 10, amplitude=amplitude)], noise=0)
    c = FlatField(np.full(raw.shape, gain), np.zeros(raw.shape))
    result = FrameProcessor(c).process(raw, roi=(240, 200, 0, 0), pixel_size=1,
                   full_scale=65535, options=AnalysisOptions(profiler=True))
    assert len(result.spots) == 1
    assert result.saturated.tolist() == [expected]
    assert ('Saturated:' in result.profiler_message) == expected


def test_calibration_metadata_rejects_wrong_camera_and_exposure(tmp_path):
    np.save(tmp_path / 'gain.npy', np.ones((3, 4)))
    np.save(tmp_path / 'dark.npy', np.zeros((3, 4)))
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(dict(gain_map='gain.npy', dark_map='dark.npy',
                                    camera_serial='123', exposure_us=29, gain_db=0, pixel_format='Mono16')))
    c = FlatField.load(path)
    camera = SimpleNamespace(serial='123', H=3, W=4, mode='16Bit', ExposureTime=29, Gain=0)
    assert c.mismatch(camera) is None
    camera.ExposureTime = 40
    assert 'exposure' in c.mismatch(camera)
    camera.serial = 'other'
    assert 'serial' in c.mismatch(camera)


def test_average_handles_large_uint16_without_overflow_and_invalid_pixels_are_masked():
    a = FrameAverage()
    for x in (60000, 20000, 40000):
        a.add(np.full((2, 2), x, np.uint16))
    np.testing.assert_allclose(a.mean, 40000)
    np.testing.assert_allclose(a.variance, 400000000)
    c = FlatField([[1, np.nan], [0, 2]], np.zeros((2, 2)))
    np.testing.assert_array_equal(c.apply(np.full((2, 2), 10), (2, 2, 0, 0)), [[10, 0], [0, 5]])
    assert c.invalid_pixels == 2
    np.testing.assert_array_equal(display_gray(np.array([[-256., 32768., 100000.]]), 65535), [[0, 128, 255]])
