import numpy as np
import scipy.integrate as integrate
import ddapp.objectmodel as om


class ControllerObj(object):

    def __init__(self, sensor, u_max=4, epsilonRand=0.3):
        self.Sensor = sensor
        self.numRays = self.Sensor.numRays
        self.actionSet = np.array([u_max,0,-u_max])
        self.epsilonRand = epsilonRand
        self.actionSetIdx = np.arange(0,np.size(self.actionSet))

    def computeControlInput(self, state, t, frame, raycastDistance=None, randomize=False):
        # test cases
        # u = 0
        # u = np.sin(t)
        if raycastDistance is None:
            self.distances = self.Sensor.raycastAll(frame)
        else:
            self.distances = raycastDistance

        # #Barry 12 controller
        

        #u = self.countStuffController()
        u, actionIdx = self.countInverseDistancesController()

        if randomize:
            if np.random.uniform(0,1,1)[0] < self.epsilonRand:
                # otherActionIdx = np.setdiff1d(self.actionSetIdx, np.array([actionIdx]))
                # randActionIdx = np.random.choice(otherActionIdx)
                actionIdx = np.random.choice(self.actionSetIdx)
                u = self.actionSet[actionIdx]

        return u, actionIdx


    def countStuffController(self):
        firstHalf = self.distances[0:self.numRays/2]
        secondHalf = self.distances[self.numRays/2:]
        tol = 1e-3;

        numLeft = np.size(np.where(firstHalf < self.Sensor.rayLength - tol))
        numRight = np.size(np.where(secondHalf < self.Sensor.rayLength - tol))

        if numLeft == numRight:
            actionIdx = 1
        elif numLeft > numRight:
            actionIdx = 2
        else:
            actionIdx = 0

        u = self.actionSet[actionIdx]
        return u, actionIdx

    def countInverseDistancesController(self):
        midpoint = np.floor(self.numRays/2.0)
        leftHalf = np.array((self.distances[0:midpoint]))
        rightHalf = np.array((self.distances[midpoint:]))
        tol = 1e-3;

        inverseLeftHalf = (1.0/leftHalf)**2
        inverseRightHalf = (1.0/rightHalf)**2

        numLeft = np.sum(inverseLeftHalf)
        numRight = np.sum(inverseRightHalf)


        if numLeft == numRight:
            actionIdx = 1
        elif numLeft > numRight:
            actionIdx = 2
        else:
            actionIdx = 0


        # print "leftHalf ", leftHalf
        # print "rightHalf", rightHalf
        # print "inverseLeftHalf", inverseLeftHalf
        # print "inverserRightHalf", inverseRightHalf
        # print "numLeft", numLeft
        # print "numRight", numRight

        u = self.actionSet[actionIdx]
        return u, actionIdx


    def computeControlInputFromFrame(self):
        carState = 0
        t = 0
        frame = om.findObjectByName('robot frame')
        return self.computeControlInput(carState, t, frame)

