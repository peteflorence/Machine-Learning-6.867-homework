__author__ = 'manuelli'
import numpy as np
from sarsa import SARSA

class SARSADiscrete(SARSA):

    def __init__(self, numInnerBins=4, numOuterBins=4, binCutoff=0.5, alphaStepSize=0.2, forceDriveStraight=False,
                 useQLearningUpdate= False, **kwargs):

        SARSA.__init__(self, alphaStepSize=0.2, **kwargs)

        self.numInnerBins=numInnerBins
        self.numOuterBins=numOuterBins
        self.numBins=numInnerBins + numOuterBins
        self.binCutoff=binCutoff
        self.useQLearningUpdate = useQLearningUpdate
        self.forceDriveStraight = forceDriveStraight
        self.initializeQValues()
        self.initializeBinData()
        self.resetElibilityTraces()
        self.eligibilityTraceThreshold = 0.1


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

    def resetElibilityTraces(self):
        #print "resetting eligibility traces to zero"
        self.eligibilityTrace = dict()


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

    def computeQValueVector(self, S):
        QVec = np.zeros(self.numActions)
        fVecShort = self.computeFeatureVector(S)
        for aIdx in xrange(self.numActions):
            fVecFull = fVecShort + (aIdx,)
            QVec[aIdx] = self.QValues[fVecFull]

        return QVec


    def computeGreedyControlPolicy(self, S, randomize=True, counter=None):
        QVec = self.computeQValueVector(S)
        actionIdx = np.argmax(QVec)

        raycastDistance = S[1]
        if self.forceDriveStraight:
            if np.min(raycastDistance) > self.sensor.rayLength - 1e-3:
                u = 0
                actionIdx = np.where(self.actionSet==u)[0][0]
                emptyQValue = False
                return u, actionIdx, emptyQValue

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


    def sarsaUpdate(self, S_current, A_idx_current, R, S_next, A_idx_next):

        featureVecCurrent = self.computeFeatureVector(S_current, A_idx = A_idx_current)
        featureVecNext = self.computeFeatureVector(S_next, A_idx=A_idx_next)



        # this does QLearning update, if this isn't specified then we do sarsa
        if self.useQLearningUpdate:
            u, actionIdx, emptyQValue = self.computeGreedyControlPolicy(S_next, randomize=False)
            featureVecNext = self.computeFeatureVector(S_next, A_idx=actionIdx)



        delta = R + self.gamma*self.QValues[featureVecNext] - self.QValues[featureVecCurrent]
        self.eligibilityTrace[featureVecCurrent] = 1.0

        QVecNext = self.computeQValueVector(S_next)
        maxIdx = np.where(QVecNext == np.max(QVecNext))[0]




        # now we perform the update, only need to do it for those that have non-zero eliglibility trace
        # which is exactly those that appear in self.eligibilityTrace
        keysToRemove = []
        for key, eVal in self.eligibilityTrace.iteritems():

            self.QValues[key] = self.QValues[key] + self.alphaStepSize*delta*eVal

            if (A_idx_next in maxIdx):
                self.eligibilityTrace[key] = self.gamma*self.lam*eVal
            else:
                self.eligibilityTrace[key] = 0

            # remove it from dict, i.e. set it to zero, if it gets too small
            if eVal < self.eligibilityTraceThreshold:
                keysToRemove.append(key)

        # if elegibility trace is sufficiently small, set it to zero
        self.deleteKeysFromDict(self.eligibilityTrace, keysToRemove)


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




