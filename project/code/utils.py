__author__ = 'manuelli'
import numpy as np

def inverseTruncate(raycastDistance, C, rayLength=None, collisionThreshold=0.0):
    if rayLength is not None:
        raycastDistance = setMaxRangeToLargeConstant(raycastDistance, rayLength, collisionThreshold=collisionThreshold)
    return np.minimum(1.0/raycastDistance, C)

def setMaxRangeToLargeConstant(raycastDistance, rayLength, collisionThreshold=0.0):
        tol=1e-3
        largeConstant = 1e5
        raycastAdjusted = raycastDistance - collisionThreshold
        maxRangeIdx = np.where(raycastDistance > rayLength - tol)
        raycastAdjusted[maxRangeIdx] = largeConstant
        return raycastAdjusted