__author__ = 'manuelli'
import numpy as np
from sarsa import SARSA

class SARSADiscrete(SARSA):

    def __init__(self, sensorObj=None, actionSet=None, gamma=0.95, lam=0.7, alphaStepSize=1e-4, epsilonGreedy=0.2,
                 cutoff=20, collisionThreshold=None, numInnerBins=5, numOuterBins=5, binCutoff=0.5):

        SARSA.__init__(self, sensorObj=sensorObj, actionSet=actionSet, gamma=gamma, lam=lam, alphaStepSize=alphaStepSize,
                       epsilonGreedy=epsilonGreedy, cutoff=cutoff, collisionThreshold=collisionThreshold)

        self.numInnerBins=numInnerBins
        self.numOuterBins=numOuterBins
        self.numBins=numInnerBins + numOuterBins
        self.binCutoff=binCutoff
        self.initializeQValues()
        self.initializeBinData()


    def initializeQValues(self):
        # we have numInnerBins + numOuterBins binary features. Order is innerBins,outerBins
        shapeTuple = (2,)*(self.numBins)

        # also need to account for the possible actions
        shapeTuple+=(np.size(self.actionSet),)

        self.QValues = np.zeros(shapeTuple)
        pass

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

    def sarsaUpdate(self, S_current, A_idx_current, R, S_next, A_idx_next):
        pass


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


