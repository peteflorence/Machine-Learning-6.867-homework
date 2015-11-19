__author__ = 'manuelli'
import numpy as np


class SARSA(object):

    def __init__(self, sensorObj=None, actionSet=None, lam=0.7, alphaStepSize=0.05, epsilonGreedy=0.1):
        if sensorObj is None or actionSet is None:
            raise ValueError("you must specify the sensorObj and the action set")
        self.numRays = sensorObj.numRays
        self.rayLength = sensorObj.rayLength
        self.lam = lam
        self.epsilonGreedy = epsilonGreedy
        self.actionSet = actionSet
        self.numActions = np.size(actionSet)
        self.tol = 1e-3
        self.alphaStepSize = alphaStepSize

        self.numFeatures = self.numRays + 1
        self.initializeWeights()


    def initializeWeights(self):
        self.weights = np.zeros((3,self.numFeatures))
        pass

    def computeFeatureVector(self, S):
        pass

    def computeSingleRayFeature(self, rayLength):
        pass

    # will probably need some helper functions inside here
    def sarsaUpdate(self, S_current, A_current, R, S_next, A_next):
        pass

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
            QVec[i] = self.computeQValue(S, A_idx, featureVector=featureVector)
        pass




