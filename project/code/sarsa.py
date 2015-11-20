__author__ = 'manuelli'
import numpy as np
import utils
import matplotlib.pyplot as plt


class SARSA(object):

    def __init__(self, sensorObj=None, actionSet=None, gamma=0.8, lam=0.7, alphaStepSize=0.05, epsilonGreedy=0.2,
                 cutoff=20, collisionThreshold=None):
        if sensorObj is None or actionSet is None or collisionThreshold is None:
            raise ValueError("you must specify the sensorObj and the action set and collisionThreshold")
        self.numRays = sensorObj.numRays
        self.rayLength = sensorObj.rayLength
        self.sensor = sensorObj
        self.lam = lam
        self.gamma = gamma # the discount factor
        self.epsilonGreedy = epsilonGreedy
        self.actionSet = actionSet
        self.numActions = np.size(actionSet)
        self.tol = 1e-3
        self.alphaStepSize = alphaStepSize
        self.cutoff = cutoff
        self.collisionThreshold = collisionThreshold

        self.numFeatures = self.numRays + 1
        self.initializeWeights()

        grid = np.arange(0, self.numRays)
        self.plotGrid = grid - np.mean(grid)


    def initializeWeights(self):
        self.weights = np.zeros((3,self.numFeatures))
        pass

    def computeFeatureVector(self, S):
        carState, raycastDistance = S
        featureVec = np.zeros(self.numFeatures)
        featureVec[0] = 1
        featureVec[1:] = utils.inverseTruncate(raycastDistance, self.cutoff, rayLength=self.rayLength,
                                               collisionThreshold=self.collisionThreshold)
        return featureVec

    def computeSingleRayFeature(self, rayLength):
        pass

    # will probably need some helper functions inside here
    # so far this is just Sarsa(0), no eligibility traces
    def sarsaUpdate(self, S_current, A_idx_current, R, S_next, A_idx_next):
        Q_vec = self.computeQValueVector(S_current)
        Q_next = self.computeQValue(S_next, A_idx_next)
        delta = R + self.gamma*Q_next - Q_vec[A_idx_current]
        gradQ = self.computeGradient(S_current, A_idx_current)

        newWeights = self.weights - self.alphaStepSize*gradQ
        self.weights = newWeights


    def computeGradient(self, S, A_idx):
        grad = np.zeros((3,self.numFeatures))
        grad[A_idx,:] = self.computeFeatureVector(S)
        return grad

    # S is a tuple S = (plantState, raycastDistance)
    def computeGreedyPolicy(self, S):
        return (u, idx)

    def computeEpsilonGreedyPolicy(self, S):
        pass

    # allow yourself to pass in a feature vector
    def computeQValue(self, S, A_idx, featureVector=None):
        if featureVector is None:
            featureVector = self.computeFeatureVector(S)

        QVal = np.dot(self.weights[A_idx,:], featureVector)
        return QVal

    def computeQValueVector(self, S):
        featureVector = self.computeFeatureVector(S)
        QVec = np.zeros(self.numActions)
        for i in xrange(0, self.numActions):
            QVec[i] = self.computeQValue(S, i, featureVector=featureVector)

        return QVec

    def plotWeights(self):


        for idx in xrange(0,self.numActions):
            plt.plot(self.plotGrid, self.weights[idx,1:], label=str(self.actionSet[idx]))

        plt.legend(loc='best')
        plt.show()

    def plotFeatureVec(self):
        carState = 0
        raycastDistance = self.sensor.raycastAllFromCurrentFrameLocation()
        featureVec = self.computeFeatureVector((carState, raycastDistance))
        plt.plot(self.plotGrid, featureVec[1:])
        plt.show()

    def computeQValueAtFrame(self, actionIdx):
        carState = 0
        raycastDistance = self.sensor.raycastAllFromCurrentFrameLocation()
        S = (carState, raycastDistance)
        return self.computeQValue(S, actionIdx)

    def computeQValueVectorAtFrame(self):
        carState = 0
        raycastDistance = self.sensor.raycastAllFromCurrentFrameLocation()
        S = (carState, raycastDistance)
        QVec = self.computeQValueVector(S)
        actionIdx = np.argmax(QVec)
        action = self.actionSet[actionIdx]
        print "QVec", QVec
        print "Best action is, ", action
        return self.computeQValueVector(S)






