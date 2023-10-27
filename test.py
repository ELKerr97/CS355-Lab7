from lab7 import world_to_camera_matrix, clip_matrix, viewport_transf_m, should_clip
import numpy as np
import math

fov = 45

def should_returnw2c_when_givenXYZandAngle():
    standard = np.array([[20], [-8*math.sqrt(2)], [-6*math.sqrt(2)], [1]])
    x = 25
    y = 20
    z = 5
    lookx = 25
    looky = 40
    lookz = 25
    point = np.array([[5],[6],[7],[1]])
    w2c = world_to_camera_matrix(x, y, z, lookx, looky, lookz)
    print(w2c)
    answer = w2c @ point
    
    if answer.all() == standard.all():
        print("SUCCESS")
    else:
        print("FAILURE")

should_returnw2c_when_givenXYZandAngle()