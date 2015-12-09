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
        #self.initializeRandomPolicyParams()
        self.initializeOkayParams()
        #self.initializeDesignedParams() # these are very good parameters

        self.mean = np.zeros((1,10))[0]
        self.epsilon = 1
        self.cov =  np.identity(10)*self.epsilon # diagonal covariance

        self.eta = 1e-1

    def initializeZeroedParams(self, random=False):
        self.leftPolicy = np.zeros((self.numRays/2,1))
        self.mirrorParams()
        self.policyTheta_0 = 0

    def initializeRandomParams(self):
        np.random.seed(4)
        self.policyTheta = np.random.randn(self.numRays,1)
        self.policyTheta_0 = 0

    def mirrorParams(self):
        self.rightPolicy = -1.0 * self.leftPolicy[ ::-1 ] # this just flips the array, and flips the sign
        self.policyTheta = np.vstack((self.leftPolicy, self.rightPolicy))

    def initializeOkayParams(self):
        self.leftPolicy = np.zeros((self.numRays/2,1))
        self.leftPolicy[:5] = -1.0 * 1
        self.leftPolicy[5:] = -1.0 * 10
        self.mirrorParams()
        self.policyTheta_0 = 0

    def initializeDesignedParams(self):
        self.leftPolicy = np.zeros((self.numRays/2,1))
        self.leftPolicy[:5] = -1.0 * 10
        self.leftPolicy[5:] = -1.0 * 100
        self.mirrorParams()
        self.policyTheta_0 = 0

    def perturbParams(self):
        self.previousLeftPolicy = self.leftPolicy*1.0

        self.w = self.sampleGaussianW()
        self.leftPolicy += self.w 
        self.mirrorParams()

    def perturbOneParam(self):
        self.previousLeftPolicy = self.leftPolicy*1.0

        randIdx = np.random.choice(len(self.leftPolicy))
        self.w = self.leftPolicy * 0.0
        
        randPerturb = np.random.randn() * 4.0
        self.w[randIdx] = randPerturb

        self.leftPolicy += self.w
        self.mirrorParams()

    def updateParams(self, reward, prevReward):
        print "######"
        print "in update"
        print "previousLeftPolicy", self.previousLeftPolicy
        print "leftPolicy", self.leftPolicy
        print "reward", reward
        print "prevReward", prevReward
        print "######"
        self.leftPolicy = self.previousLeftPolicy + self.eta * (reward - prevReward) * self.w
        self.mirrorParams()

    def revertParams(self):
        self.leftPolicy = self.previousLeftPolicy
        self.mirrorParams()

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






