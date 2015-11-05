import numpy as np
import math
from LineCircleIntersect import intersect

def computeLaserDepthsTraj(xworld,yworld,laseAngles,S_maxdist,obsField):
  
    # first index is step in time
    # second index is the laser depth as a function of laseAngle
    laserDepthsTraj = np.zeros((len(xworld),len(laseAngles)))

    x_laser_endpointTraj = np.zeros((len(xworld),len(laseAngles)))
    y_laser_endpointTraj = np.zeros((len(xworld),len(laseAngles)))

    for i in range(len(xworld)):
        for lasenum in range(len(laseAngles)):
            x_laser_endpointTraj[i,lasenum] = S_maxdist*math.cos(laseAngles[lasenum])
            y_laser_endpointTraj[i,lasenum] = S_maxdist*math.sin(laseAngles[lasenum])
            origin = np.array((0,0))
            laser_endpoint = np.array((x_laser_endpointTraj[i,lasenum],y_laser_endpointTraj[i,lasenum]))
            for obs in obsField.ObstaclesList:
                Q = np.array((obs.xtraj[i],obs.ytraj[i]))
                r = obs.radius
                _, pt = intersect(origin,laser_endpoint,Q,r)
                if pt is not None:
                    if (pt[0]**2 + pt[1]**2 < x_laser_endpointTraj[i,lasenum]**2 + y_laser_endpointTraj[i,lasenum]**2):
                        x_laser_endpointTraj[i,lasenum] = pt[0]
                        y_laser_endpointTraj[i,lasenum] = pt[1]

            depth = math.sqrt(x_laser_endpointTraj[i,lasenum]**2 + y_laser_endpointTraj[i,lasenum]**2)
            laserDepthsTraj[i,lasenum] = depth

    return laserDepthsTraj, x_laser_endpointTraj, y_laser_endpointTraj
