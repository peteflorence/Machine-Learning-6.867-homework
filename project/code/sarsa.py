__author__ = 'manuelli'
import numpy as np

class SARSA(object):

    def __init__(self, sensorObj=None, actionSet=None, gamma=0.95, lam=0.7, alphaStepSize=1e-4, epsilonGreedy=0.2,
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


    def sarsaUpdate(self, S_current, A_idx_current, R, S_next, A_idx_next):
        raise ValueError("subclass must implement this method")


