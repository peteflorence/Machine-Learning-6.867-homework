import numpy as np
import scipy.integrate as integrate

class ControllerObj(object):

    def __init__(self, sensor):
        self.Sensor = sensor

    def computeControlInput(self, state, t, frame):
        #u = 0
        u = np.sin(t)
        intersections = self.Sensor.raycastAll(frame)
        return u

    #     #Barry 12 controller
    #     c_1 = 1
    #     c_2 = 10
    #     c_3 = 100

    #     F = rays*0.0
    #     for i in range(0,numRays):
    #         F[i] = -c_1*