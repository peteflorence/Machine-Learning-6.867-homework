__author__ = 'peteflorence'
import numpy as np
from policySearch import PolicySearch

class PolicySearchFiniteDifferences(PolicySearch):

    def __init__(self, numInnerBins=4, numOuterBins=4, binCutoff=0.5, alphaStepSize=0.2,
                 useQLearningUpdate= False, **kwargs):

        PolicySearch.__init__(self, alphaStepSize=0.2, **kwargs)

        self.numInnerBins=numInnerBins
        self.numOuterBins=numOuterBins
        self.numBins=numInnerBins + numOuterBins
        self.binCutoff=binCutoff
        self.initializeBinData()

        self.initializePolicyParams()

    def initializePolicyParams(self):
        np.random.seed(4)
        self.policyTheta = np.random.randn(self.numRays,1)

    def computeControlPolicy(self, S):
        raycastDistance = S[1]
        print "raycastDistance shape", np.shape(raycastDistance)
        u = np.dot(policyTheta.T, raycastDistance)
        print "dot product is", u

        return u

    #Kiefer-Wolfowitz procedure, change one theta at a time
    def finiteDifferenceGradientEstimator(self):
        zeroVector = self.policyTheta * 0.0
        
        maxIter = 1000
        numIter = 0

        perturbationSize = 1e-2

        for i in xrange(maxIter):
            
            parameterToAdjust = np.random.choice(20, 1)[0] # this chooses a random number in range[0,19]
            
            policyVariation = zeroVector
            policyVariation[parameterToAdjust] = perturbationSize

            # then need to do roll-outs


    def computeDummyControlPolicy(self, S, randomize=True, counter=None):
        actionIdx = 1                  # hardcoded to go straight, for debugging
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

        return u, actionIdx

    def policySearchUpdate(self, S_current, A_idx_current, R, S_next, A_idx_next):
        self.updated = True
        

    def initializeBinData(self):
        self.binData = ()

        innerBinRays = self.computeBinRayIdx(self.numInnerBins)
        for idx in xrange(self.numInnerBins):
            d = dict()
            d['type'] = "inner"
            d['idx'] = idx
            d['rayIdx'] = innerBinRays[idx]
            d['minRange'] = self.collisionThreshold
            d['maxRange'] = self.collisionThreshold + (self.rayLength - self.collisionThreshold)*self.binCutoff
            self.binData+=(d,)

        outerBinRays = self.computeBinRayIdx(self.numOuterBins)
        for idx in xrange(self.numOuterBins):
            d = dict()
            d['type'] = "outer"
            d['idx'] = idx
            d['rayIdx'] = innerBinRays[idx]
            d['minRange'] = self.collisionThreshold + (self.rayLength - self.collisionThreshold)*self.binCutoff
            d['maxRange'] = self.rayLength - self.tol
            self.binData+=(d,)


    def computeBinRayIdx(self, numBins):
        if numBins==0:
            return 0
        binRays = ()
        cutoffs = np.floor(np.linspace(0,self.numRays,numBins+1))
        for idx in xrange(numBins):
            rayLeftIdx = cutoffs[idx]
            rayRightIdx = cutoffs[idx+1]-1
            if idx==numBins-1:
                rayRightIdx=self.numRays
            binRays+=(np.arange(rayLeftIdx,rayRightIdx, dtype='int'),)

        return binRays

    def computeBinOccupied(self,raycastDistance, binNum):
        occupied=0
        minRange = self.binData[binNum]['minRange']
        maxRange = self.binData[binNum]['maxRange']
        rayIdx = self.binData[binNum]['rayIdx']
        if (np.any(np.logical_and(raycastDistance[rayIdx] > minRange, raycastDistance[rayIdx] < maxRange) ) ):
            occupied = 1

        return occupied


    def computeFeatureVector(self,S,A_idx=None):
        raycastDistance = S[1]
        featureTuple = ()
        for idx in xrange(self.numBins):
            featureTuple+= (self.computeBinOccupied(raycastDistance, idx),)


        if A_idx is not None:
            featureTuple += (A_idx,)

        return featureTuple


    def deleteKeysFromDict(self, d, keys):
        for key in keys:
            if key in d:
                del d[key]



    def computeFeatureVectorFromCurrentFrame(self):
        raycastDistance = self.sensor.raycastAllFromCurrentFrameLocation()
        carState = 0
        S = (carState, raycastDistance)
        featureVec = self.computeFeatureVector(S)
        print ""
        print "innerBins ", featureVec[:self.numInnerBins]
        print "outerBins", featureVec[self.numInnerBins:]
        print ""
        return featureVec


    def computeQValueVectorFromCurrentFrame(self):
        QVec = np.zeros(3)
        fVec = self.computeFeatureVectorFromCurrentFrame()
        for aIdx in xrange(self.numActions):
            fVecTemp = fVec + (aIdx,)
            QVec[aIdx] = self.QValues[fVecTemp]


        print "QVec is", QVec
        aIdxMax = np.argmax(QVec)
        if QVec[aIdxMax] == 0.0:
            print "table value never updated"
        else:
            print "best action is", aIdxMax




