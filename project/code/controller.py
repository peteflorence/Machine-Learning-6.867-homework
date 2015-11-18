import numpy as np
import scipy.integrate as integrate

class ControllerObj(object):

    def __init__(self, sensor):
        self.Sensor = sensor

    def computeControlInput(self, state, t, frame):
        # test cases
        # u = 0
        # u = np.sin(t)
        distances = self.Sensor.raycastAll(frame)



        # #Barry 12 controller
        numRays = self.Sensor.numRays
        # rays = self.Sensor.rays

        # c_1 = 1
        # c_2 = 1
        # c_3 = 1

        # F = distances*0.0

        # for i in range(0,numRays):
        #     r = distances[i]
        #     if r is not None:
        #         r_hat = rays[:,i]
        #         x_hat = np.array(frame.transform.TransformNormal(rays[:,numRays/2]))

        #         G_hat = x_hat

        #         F[i] = np.dot(-c_1 * x_hat, r_hat * (1.0 - (r/c_3))**2.0 / (1.0 + (r/c_3)**3.0))

        # Ftot = np.sum(F)
        # F_G = np.dot(-c_2 * G_hat, x_hat)

        # u = F_G + Ftot
        # print u


        # Count stuff controller
        firstHalf = distances[0:numRays/2]
        secondHalf = distances[numRays/2:]

        numLeft = np.count_nonzero(firstHalf)
        numRight = np.count_nonzero(secondHalf)

        if numLeft == numRight:
            u = 0
        elif numLeft > numRight:
            u = 1
        else:
            u = -1

        return u