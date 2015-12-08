__author__ = 'peteflorence'
import numpy as np
from policySearch import PolicySearch

class PolicySearchSGD(PolicySearch):

    def __init__(self, numInnerBins=4, numOuterBins=4, binCutoff=0.5, alphaStepSize=0.2,
                 useQLearningUpdate= False, **kwargs):

        PolicySearch.__init__(self, alphaStepSize=0.2, **kwargs)

        self.numInnerBins=numInnerBins
        self.numOuterBins=numOuterBins
        self.numBins=numInnerBins + numOuterBins
        self.binCutoff=binCutoff
        #self.initializeZeroedParams()
        self.initializePolicyParams() #this doesn't crash

        self.mean = np.zeros((1,20))[0]
        self.epsilon = 1e-3
        self.cov =  np.identity(20)*self.epsilon # diagonal covariance

    def initializeZeroedParams(self, random=False):
        self.policyTheta = np.zeros((self.numRays,1))
        self.policyTheta_0 = 0

    def initializePolicyParams(self, random=False):
        if random==True:
            np.random.seed(4)
            self.policyTheta = np.random.randn(self.numRays,1)
        else:
            self.policyTheta = np.zeros((self.numRays,1))
            self.policyTheta[:len(self.policyTheta)/2] = -1.0 * 100
            self.policyTheta[len(self.policyTheta)/2:] = 1.0 * 100
            self.policyTheta[0:4] = 0.0
            self.policyTheta[-4:] = 0.0
            self.policyTheta_0 = 0


    def sampleGaussianW(self):
        w = np.random.multivariate_normal(self.mean, self.cov).T
        return np.matrix(w).T

    def computeSigmoidedControlPolicy(self, S):
        raycastDistance = S[1]
        feature = raycastDistance * 0.0

        for idx, i in enumerate(feature):
            feature[idx] = (1/raycastDistance[idx]) **2

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        a = np.dot(self.policyTheta.T, feature) + self.policyTheta_0
        u = (sigmoid(a) - 0.5)*8
        return u, 0


    def computeDummyControlPolicy(self, S, randomize=True, counter=None):
        actionIdx = 1                  # hardcoded to go straight, for debugging
        u = self.actionSet[actionIdx]
        return u, actionIdx


    def computeFeatureVector(self, S):
        carState, raycastDistance = S
        featureVec = np.zeros(self.numFeatures)
        featureVec[0] = 1
        featureVec[1:] = utils.inverseTruncate(raycastDistance, self.cutoff, rayLength=self.rayLength,
                                               collisionThreshold=self.collisionThreshold)

        return featureVec






