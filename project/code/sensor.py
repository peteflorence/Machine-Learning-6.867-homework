import ddapp.vtkAll as vtk

import numpy as np
import ddapp.objectmodel as om


class SensorObj(object):

    def __init__(self, FOV=180.0, numRays=20, rayLength=8):
        self.numRays = numRays
        self.rayLength = rayLength

        FOVrad = FOV * np.pi/180.0
        self.angleMin = -FOVrad/2
        self.angleMax = FOVrad/2

        self.angleGrid = np.linspace(self.angleMin, self.angleMax, self.numRays)

        self.rays = np.zeros((3,self.numRays))
        self.rays[0,:] = np.cos(self.angleGrid)
        self.rays[1,:] = np.sin(self.angleGrid)

    def setLocator(self, locator):
        self.locator = locator

    def raycastAll(self,frame):

        distances = np.zeros(self.numRays)

        origin = np.array(frame.transform.GetPosition())

        for i in range(0,self.numRays):
            ray = self.rays[:,i]
            rayTransformed = np.array(frame.transform.TransformNormal(ray))
            intersection = self.raycast(self.locator, origin, origin + rayTransformed*self.rayLength)
            if intersection is None:
                distances[i] = self.rayLength
            else:
                distances[i] = np.linalg.norm(intersection - origin)

        return distances

    def raycastAllFromCurrentFrameLocation(self):
        frame = om.findObjectByName('robot frame')
        return self.raycastAll(frame)


    def raycast(self, locator, rayOrigin, rayEnd):

        tolerance = 0.0 # intersection tolerance
        pt = [0.0, 0.0, 0.0] # data coordinate where intersection occurs
        lineT = vtk.mutable(0.0) # parametric distance along line segment where intersection occurs
        pcoords = [0.0, 0.0, 0.0] # parametric location within cell (triangle) where intersection occurs
        subId = vtk.mutable(0) # sub id of cell intersection

        result = locator.IntersectWithLine(rayOrigin, rayEnd, tolerance, lineT, pt, pcoords, subId)

        return pt if result else None