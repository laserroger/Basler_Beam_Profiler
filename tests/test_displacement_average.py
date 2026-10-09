import numpy as np
from beam_profiler.processing.displacement import DisplacementAverage


def data(value=0., reverse=False):
    order=np.array([1,0]) if reverse else np.array([0,1])
    return {'valid':True,'reference':np.array([[10.,20.],[30.,40.]])[order],
            'residual':np.array([[value,0.],[value+10,0.]])[order],
            'rows':np.array([0,0])[order],'columns':np.array([0,1])[order]}


def test_five_frame_mean_matches_cells_and_rolls():
    avg=DisplacementAverage()
    for i in range(4):
        result=avg.update(data(i, bool(i%2)),('roi',),i)
        assert not result['valid']
    result=avg.update(data(4),('roi',),4)
    assert result['averaged_frames']==5
    np.testing.assert_allclose(result['residual'],[[2,0],[12,0]])
    assert avg.update(data(100),('roi',),4) is result
    result2=avg.update(data(5),('roi',),5)
    np.testing.assert_allclose(result2['residual'],[[3,0],[13,0]])
    np.testing.assert_allclose(result['residual'],[[2,0],[12,0]])


def test_roi_invalid_frame_and_grid_change_reset():
    avg=DisplacementAverage()
    for i in range(5):avg.update(data(i),'old ROI',i)
    assert not avg.update(data(10),'new ROI',6)['valid']
    assert len(avg.frames)==1
    assert not avg.update({'valid':False,'reason':'Saturated'},'new ROI',7)['valid']
    assert len(avg.frames)==0
    for i in range(5):avg.update(data(i),'new ROI',i+10)
    changed=data();changed['columns']=np.array([1,2])
    assert not avg.update(changed,'new ROI',20)['valid']
    assert len(avg.frames)==1
    avg.reset()
    assert not avg.update(data(),'new ROI',21)['valid']
