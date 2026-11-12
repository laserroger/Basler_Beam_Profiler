from dataclasses import replace
import numpy as np
import pytest
from beam_profiler.synthetic import Spot, render
from beam_profiler.processing.pipeline import FrameProcessor, AnalysisOptions
from beam_profiler.processing.sizes import spot_sizes, SizeMonitor
from beam_profiler.processing.spots import SpotArray
from beam_profiler.server import create_app
from beam_profiler.session import LiveSession, SessionSnapshot
POSITIONS = ((100,100), (1000,1000), (100,1000))


def fitted():
    frame=render(200,160,[Spot(100,80,7,4,theta=.6,amplitude=.6)],noise=0)
    return FrameProcessor().process(frame,roi=(200,160,10,20),pixel_size=2.5e-6,
        full_scale=65535,options=AnalysisOptions(fitting=True))


def test_diameter_matches_rotated_gaussian_and_pixel_calibration():
    data=spot_sizes(fitted())
    assert data['spot_count']==data['valid_count']==1
    assert data['major_um'][0] == pytest.approx(4*7*2.5,rel=.02)
    assert data['minor_um'][0] == pytest.approx(4*4*2.5,rel=.02)
    assert data['x_sensor_px'][0] == pytest.approx(110,abs=.01)
    assert data['y_sensor_px'][0] == pytest.approx(100,abs=.01)
    assert not any('mean' in key for key in data)


def test_empty_saturated_and_invalid_fit_are_gaps():
    r=fitted();m=SizeMonitor();m.update(r);assert m.select((110,100))
    for bad in (replace(r,saturated=np.ones(1,dtype=bool)),
                replace(r,spots=SpotArray.empty(),saturated=np.zeros(0,dtype=bool))):
        data=m.update(bad)
        assert data['valid_count']==0
        assert data['traces'][-1]['major_um'] is None
        assert data['traces'][-1]['history'][-1]['major_um'] is None
    r.spots.sigma_0[0]=np.nan
    assert m.update(r)['traces'][-1]['major_um'] is None


def test_history_expiry_context_reset_and_snapshot_ownership():
    r=fitted();m=SizeMonitor();m.update(r);m.select((110,100))
    old=m.update(replace(r,timestamp=0),'camera1')
    new=m.update(replace(r,timestamp=61),'camera1')
    assert len(new['traces'][-1]['history'])==1
    assert old['traces'][-1]['history'][0]['timestamp']==0
    assert len(m.update(replace(r,timestamp=62),'camera2')['traces'][-1]['history'])==1


def test_sizes_http_and_legacy_radius_compatibility():
    session=LiveSession();client=create_app(session).test_client()
    assert client.get('/api/spot_sizes').status_code==503
    r=fitted();m=SizeMonitor();data=m.update(r)
    assert data['traces']==[]
    session.publish(SessionSnapshot(result=r,sizes=data))
    assert client.get('/api/spot_sizes').json==data
    assert client.get('/sizes').status_code==200
    assert client.get('/api/spot_sizes/plot').mimetype=='image/png'
    spot=client.get('/api/spots').json['spots'][0]
    assert spot['diameter_major_um']==pytest.approx(2*spot['sigma_0']*2.5)
    assert client.post('/api/spot_sizes/select',json={'position':[110,100]}).status_code==202
    assert session.take_commands()==[('trace_spot',[110.,100.])]
    for position in ([0,0],[float('inf'),100],[1], 'missing'):
        assert client.post('/api/spot_sizes/select',json={'position':position}).status_code==400
    assert client.post('/api/spot_sizes/select',json={'position':None}).status_code==202


def test_trace_follows_position_across_reordering_and_does_not_jump():
    frame=render(1100,1100,[Spot(x,y,5,amplitude=.6) for x,y in POSITIONS],noise=0)
    r=FrameProcessor().process(frame,roi=(1100,1100,0,0),pixel_size=2.5e-6,
        full_scale=65535,options=AnalysisOptions(fitting=True))
    m=SizeMonitor();m.update(r);assert m.select((1000,1000))
    d=m.update(replace(r,spots=r.spots[::-1],saturated=r.saturated[::-1]))
    assert d['traces'][-1]['x_sensor_px']==pytest.approx(1000,abs=.01)
    # Losing the selected spot must not attach its trace to another corner.
    keep=(r.spots.x<500)
    d=m.update(replace(r,spots=r.spots[keep],saturated=r.saturated[keep]))
    assert not d['traces'][-1]['valid']
    assert d['traces'][-1]['history'][-1]['major_um'] is None
    assert m.select((100,100))
    assert len(m.update(r)['traces'][-1]['history'])==1
    assert m.select(None)
    assert m.update(r)['traces']==[]


def test_ambiguous_candidates_make_a_gap():
    r=fitted();m=SizeMonitor();m.update(r);m.select((110,100))
    duplicated=SpotArray.concat([r.spots,r.spots])
    d=m.update(replace(r,spots=duplicated,saturated=np.zeros(2,dtype=bool)))
    assert d['traces'][-1]['reason']=='ambiguous detections'
    assert d['traces'][-1]['history'][-1]['major_um'] is None


def test_native_click_selects_sensor_coordinates_without_starting_a_rectangle():
    import cv2
    from types import SimpleNamespace
    from beam_profiler.ui.viewer import Viewer
    r=fitted();m=SizeMonitor();m.update(r)
    viewer=Viewer.__new__(Viewer)
    viewer.camera=SimpleNamespace(ROI=r.roi)
    viewer.show_sizes=True
    viewer.size_monitor=m
    viewer.rect_disp=viewer.fit_rect_disp=None
    x,y=viewer.view.to_display(110,100)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN,x,y,0,None)
    target=next(iter(m.targets.values()))
    assert (target['x_sensor_px'],target['y_sensor_px'])==pytest.approx((110,100),abs=.01)
    assert viewer.rect_disp is None


def three_spots():
    frame=render(1100,1100,[Spot(x,y,5,amplitude=.6) for x,y in POSITIONS],noise=0)
    return FrameProcessor().process(frame,roi=(1100,1100,0,0),pixel_size=2.5e-6,
        full_scale=65535,options=AnalysisOptions(fitting=True))


def test_multiselect_preserves_each_history_color_and_id():
    r=three_spots();m=SizeMonitor();m.update(r)
    assert m.select(POSITIONS[0])
    first=m.update(replace(r,timestamp=0))['traces'][0]
    assert m.select(POSITIONS[1])
    data=m.update(replace(r,timestamp=1))
    assert len(data['traces'])==2
    assert len(data['traces'][0]['history'])==2
    assert len(data['traces'][1]['history'])==1
    assert data['traces'][0]['color']==first['color']
    assert m.select(POSITIONS[2])
    data=m.update(replace(r,timestamp=2))
    assert len(data['traces'])==3
    ids=[t['id'] for t in data['traces']]
    colors=[t['color'] for t in data['traces']]
    assert len(set(colors))==3
    assert m.select(POSITIONS[1])  # remove only the middle trace
    data=m.update(replace(r,timestamp=3))
    assert [t['id'] for t in data['traces']]==[ids[0],ids[2]]
    assert [t['color'] for t in data['traces']]==[colors[0],colors[2]]
    assert len(data['traces'][0]['history'])==4
    assert len(data['traces'][1]['history'])==2
    # Previously published histories belong to their original snapshots.
    assert len(first['history'])==1
    m.select(None)
    assert m.update(r)['traces']==[]


def test_one_lost_or_saturated_beam_does_not_break_other_traces():
    r=three_spots();m=SizeMonitor();m.update(r)
    for position in POSITIONS:
        assert m.select(position)
    keep=r.spots.x<500
    saturated=np.zeros(keep.sum(),dtype=bool)
    saturated[np.argmin(r.spots[keep].y)]=True
    d=m.update(replace(r,spots=r.spots[keep],saturated=saturated))
    assert [t['valid'] for t in d['traces']]==[False,False,True]
    assert [t['history'][-1]['major_um'] is None for t in d['traces']]==[True,True,False]
    assert m.select(POSITIONS[1])  # deselect a missing beam by its ring
    assert len(m.targets)==2


def test_nearby_beams_can_be_selected_independently():
    r=fitted();left=r.spots
    right=SpotArray(*(getattr(left,key).copy() for key in
                     ('x','y','sigma_0','sigma_1','vec_0','vec_1','I0','I0_weighted')))
    right.x+=20
    r=replace(r,spots=SpotArray.concat([left,right]),saturated=np.zeros(2,dtype=bool))
    m=SizeMonitor();m.update(r)
    assert m.select((110,100))
    assert m.select((130,100))
    data=m.update(r)
    assert len(data['traces'])==2
    assert all(t['valid'] for t in data['traces'])
    assert all(t['gate_px']<10 for t in data['traces'])


def test_reset_time_keeps_selected_spots_and_starts_new_histories_at_zero():
    r=three_spots();m=SizeMonitor();m.update(r)
    for position in POSITIONS:
        m.select(position)
    before=m.update(replace(r,timestamp=100))
    m.update(replace(r,timestamp=110))
    m.reset_time()
    after=m.update(replace(r,timestamp=120))
    assert after['time_origin']==120
    assert [t['id'] for t in after['traces']]==[t['id'] for t in before['traces']]
    assert all(len(t['history'])==1 for t in after['traces'])
    assert before['time_origin']==100
    assert before['traces'][0]['history'][0]['timestamp']==100


def test_time_window_prunes_history_without_resetting_clock_or_selections():
    r=fitted();m=SizeMonitor();m.update(r);m.select((110,100))
    m.update(replace(r,timestamp=0))
    m.update(replace(r,timestamp=10))
    m.set_window(5)
    d=m.update(replace(r,timestamp=11))
    assert [h['timestamp'] for h in d['traces'][0]['history']]==[10,11]
    assert d['time_origin']==0
    assert d['window_seconds']==5
    for bad in (0,-1,601,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            m.set_window(bad)


def test_plot_controls_validate_ranges_and_queue_reset_and_window():
    r=fitted();m=SizeMonitor();m.update(r);m.select((110,100))
    session=LiveSession()
    session.publish(SessionSnapshot(result=r,sizes=m.update(r)))
    client=create_app(session).test_client()
    for args in ('?y_min=49&y_max=55','?y_min=0&y_max=1'):
        assert client.get('/api/spot_sizes/plot'+args).status_code==200
    for args in ('?y_min=55&y_max=49','?y_min=nan&y_max=55','?y_min=1','?y_min=-1&y_max=5'):
        assert client.get('/api/spot_sizes/plot'+args).status_code==400
    assert client.post('/api/spot_sizes/window',json={'seconds':10}).status_code==202
    assert client.post('/api/spot_sizes/reset_time').status_code==202
    for name,value in session.take_commands():
        if name=='trace_window':
            m.set_window(value)
        elif name=='trace_reset':
            m.reset_time()
    session.publish(SessionSnapshot(result=r,sizes=m.update(r)))
    d=client.get('/api/spot_sizes').json
    assert d['window_seconds']==10
    assert d['time_origin']==d['timestamp']
    assert len(d['traces'])==1 and len(d['traces'][0]['history'])==1
    for bad in ({},{'seconds':0},{'seconds':601},{'seconds':True},{'seconds':'bad'}):
        assert client.post('/api/spot_sizes/window',json=bad).status_code==400


def test_shared_history_memory_limit_drops_oldest_without_losing_selections(monkeypatch):
    from beam_profiler.processing import sizes
    monkeypatch.setattr(sizes,'HISTORY_STORAGE_BYTES',12_000)
    monkeypatch.setattr(sizes,'MAX_HISTORY_SAMPLES',60)
    r=three_spots();m=SizeMonitor();m.update(r);m.set_window(600)
    for position in POSITIONS:
        m.select(position)
    for i in range(100):
        d=m.update(replace(r,timestamp=i*.1))
        assert d['history_storage_bytes']<=12_000
        assert sum(len(t['history']) for t in d['traces'])<=60
    assert len(d['traces'])==3
    assert d['time_origin']==0
    assert all(t['history'][0]['timestamp']>0 for t in d['traces'])
    m.select(POSITIONS[1])
    assert m._sample_bytes>=0
    m.reset_time()
    assert m._sample_bytes==0
