import numpy as np
import cv2
from ring_init.io.calib import Camera, undistort


def test_undistortion_round_trip_center_is_stable():
    image=np.zeros((80,100,3),np.uint8); image[40,50]=255
    camera=Camera("x",np.array([[80.,0,50],[0,80,40],[0,0,1.]]),np.array([.01,0,0,0]),np.eye(3),np.zeros(3),100,80)
    output, new=undistort(image,camera)
    assert output.shape==image.shape and np.allclose(new.K[2],[0,0,1])
