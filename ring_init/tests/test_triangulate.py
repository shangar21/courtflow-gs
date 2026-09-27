import numpy as np
from ring_init.io.calib import Camera
from ring_init.geom.triangulate import triangulate_dlt, filter_pair


def camera(center):
    K=np.array([[800.,0,320],[0,800,240],[0,0,1.]])
    R=np.eye(3); return Camera("x",K,np.zeros(0),R,-np.asarray(center,float),640,480)


def test_dlt_known_cameras_recovers_metric_point():
    a,b=camera([0,0,0]),camera([1,0,0]); point=np.array([[.2,.1,3.]])
    ua,_=a.project(point); ub,_=b.project(point)
    recovered=triangulate_dlt(ua,ub,a,b)
    assert np.max(np.abs(recovered-point))<1e-4
    assert filter_pair(recovered,ua,ub,a,b,1.5,3.).item()
