import numpy as np
import scipy.integrate as integrate

class ControllerObj(object):

    def __init__(self, sensor):
        self.Sensor = sensor

    def computeControlInput(self, state, t, frame):
        # base tests
        # u = 0
        # u = np.sin(t)
        

        intersections = self.Sensor.raycastAll(frame)
        u = 0
        
        # #Barry 12 controller
        # c_1 = 1
        # c_2 = 10
        # c_3 = 100

        # F = intersections*0.0
        # for idx, value in enumerate(intersections):
        #     F[i] = -c_1*


        # return u