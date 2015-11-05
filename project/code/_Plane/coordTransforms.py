import math
import numpy as np

def transformStandardPolarTheta_to_PlaneTheta(angle):
    if isinstance(angle, np.ndarray):
      x = np.cos(angle)
      y = np.sin(angle)
      thetatop = np.arcsin(x)
      theta = np.arcsin(x)
      for i in angle:
        if y[i] < 0 :
          if x[i] > 0:
            theta[i] = math.pi - thetatop[i]
          else:
            theta[i] = -math.pi - thetatop[i]
        else:
          theta[i] = thetatop[i]


    else:
      x = np.cos(angle)
      y = np.sin(angle)
      thetatop = np.arcsin(x)
      if y < 0 :
        if x > 0:
          theta = math.pi - thetatop
        else:
          theta = -math.pi - thetatop
      else:
        theta = thetatop
    
    return theta