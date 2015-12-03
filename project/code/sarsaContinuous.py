__author__ = 'manuelli'
import numpy as np
import utils
import matplotlib.pyplot as plt
from sarsa import SARSA

class SARSAContinuous(SARSA):

    def __init__(self, alphaStepSize=1e-4, **kwargs):

        SARSA.__init__(self, alphaStepSize=1e-4, **kwargs)


        self.numFeatures = self.numRays + 1
        self.initializeWeights()
        self.resetElibilityTraces()

        grid = np.arange(0, self.numRays)
        self.plotGrid = grid - np.mean(grid)


    def initializeWeights(self):
        self.weights = np.zeros((self.numActions,self.numFeatures))

    def resetElibilityTraces(self):
        self.eligibilityTraces = np.zeros((self.numActions, self.numFeatures))

    def computeFeatureVector(self, S):
        carState, raycastDistance = S
        featureVec = np.zeros(self.numFeatures)
        featureVec[0] = 1
        featureVec[1:] = utils.inverseTruncate(raycastDistance, self.cutoff, rayLength=self.rayLength,
                                               collisionThreshold=self.collisionThreshold)

        return featureVec


    # implements the general SARSA(lambda) update using function approximation
    def sarsaUpdate(self, S_current, A_idx_current, R, S_next, A_idx_next):
        Q_vec = self.computeQValueVector(S_current)
        Q_next = self.computeQValue(S_next, A_idx_next)
        delta = R + self.gamma*Q_next - Q_vec[A_idx_current]
        gradQ = self.computeGradient(S_current, A_idx_current)

        # need to update the eligibility traces
        self.eligibilityTraces = self.gamma*self.lam*self.eligibilityTraces + gradQ
        newWeights = self.weights + self.alphaStepSize*delta*self.eligibilityTraces

        self.weights = newWeights


    def computeGradient(self, S, A_idx):


        # if self.useSumFeature:
        #     featureList = self.computeFeatureVector(S)
        #     grad = np.zeros((3,self.numFeatures))
        #     grad[A_idx,:] = featureList[1]
        #     grad = (featureList[0], grad)
        # else:

        # this is for linear function approximation
        grad = np.zeros((self.numActions, self.numFeatures))
        grad[A_idx,:] = self.computeFeatureVector(S)
        return grad


    def computeGreedyControlPolicy(self, S, randomize=True, counter=None):
        QVec = self.computeQValueVector(S)
        actionIdx = np.argmax(QVec)

        if QVec[actionIdx] == 0.0:
            emptyQValue=True
        else:
            emptyQValue=False


        u = self.actionSet[actionIdx]

        if randomize:
            if counter is not None:
                epsilon = self.epsilonGreedyDecay(counter)
            else:
                epsilon = self.epsilonGreedy

            if np.random.uniform(0,1,1)[0] < epsilon:
                # otherActionIdx = np.setdiff1d(self.actionSetIdx, np.array([actionIdx]))
                # randActionIdx = np.random.choice(otherActionIdx)
                actionIdx = np.random.choice(self.actionSetIdx)
                u = self.actionSet[actionIdx]

        return u, actionIdx, emptyQValue


    # allow yourself to pass in a feature vector
    def computeQValue(self, S, A_idx, featureVector=None):
        if featureVector is None:
            featureVector = self.computeFeatureVector(S)

        # if self.useSumFeature:
        #     QVal = self.weights[0]*featureVector[0]
        #     QVal += np.dot(self.weights[1][A_idx,:], featureVector[1])
        # else:

        QVal = np.dot(self.weights[A_idx,:], featureVector)
        return QVal

    def computeQValueVector(self, S):
        featureVector = self.computeFeatureVector(S)
        QVec = np.zeros(self.numActions)
        for i in xrange(0, self.numActions):
            QVec[i] = self.computeQValue(S, i, featureVector=featureVector)

        return QVec

    def plotWeights(self):

        weights = self.weights

        for idx in xrange(0,self.numActions):
            plt.plot(self.plotGrid, weights[idx,1:], label=str(self.actionSet[idx]))

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






