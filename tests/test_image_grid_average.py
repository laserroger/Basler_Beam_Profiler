import numpy as np
from beam_profiler import fitconfig
from beam_profiler.processing import displacement


def test_averages_pixels_before_fitting_and_resets_on_exposure(monkeypatch):
    fitted = []
    def fit(mean, *args):
        fitted.append(mean.copy())
        return {'valid': True}, None
    monkeypatch.setattr(displacement, 'fit_averaged_grid', fit)
    avg = displacement.ImageGridAverage()
    options = dict(roi=(4, 3, 0, 0), camera='test', exposure=100,
                   gain=0, full_scale=65535, config=fitconfig.active())
    for i in range(19):
        assert not avg.update(np.full((3, 4), i, dtype=np.uint16), frame_id=i, **options)['valid']
    assert not fitted
    result = avg.update(np.full((3, 4), 19, dtype=np.uint16), frame_id=19, **options)
    assert result['averaged_frames'] == 20
    np.testing.assert_array_equal(fitted[0], np.full((3, 4), 9.5))
    avg.update(np.zeros((3, 4)), frame_id=19, **options)
    assert avg.count == 0
    options['exposure'] = 200
    assert not avg.update(np.zeros((3, 4)), frame_id=20, **options)['valid']
    assert avg.count == 1
    avg.reset()
    assert avg.total is None and avg.spots is None and not avg.latest['valid']


def test_averaged_image_fits_grid_and_quadratic_bow():
    yy, xx = np.mgrid[:220, :220]
    frame = np.zeros((220, 220), dtype=float)
    for y in np.linspace(30, 190, 5):
        for x in np.linspace(30, 190, 5):
            bow = .15 * ((y-110)/80)**2
            frame += 30000*np.exp(-((xx-x-bow)**2+(yy-y)**2)/8)
    result, spots = displacement.fit_averaged_grid(
        frame, (220,220,0,0), 65535, fitconfig.active())
    assert result['valid'] and len(spots) == 25
    assert result['rms_px'] > .03
    assert result['fit_error_rms_px'] < .01
