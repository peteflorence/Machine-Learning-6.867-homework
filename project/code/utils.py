__author__ = 'manuelli'
import numpy as np

def inverseTruncate(raycastDistance, C):
    return np.minimum(1.0/raycastDistance, C)