import ddapp.vtkAll as vtk

import numpy as np

class SensorObj(object):

    def __init__(self, FOV=180.0, numRays=20, rayLength=5):
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

        intersections = np.zeros((3,self.numRays))

        origin = np.array(frame.transform.GetPosition())

        for i in range(0,self.numRays):
            ray = self.rays[:,i]
            rayTransformed = np.array(frame.transform.TransformNormal(ray))
            intersections[:,i] = self.raycast(self.locator, origin, origin + rayTransformed*self.rayLength)
            print "Intersection is", intersections[:,i]

            #distance = np.linalg.norm(intersections[:,i] - origin)

        return intersections


    def raycast(self, locator, rayOrigin, rayEnd):

        tolerance = 0.0 # intersection tolerance
        pt = [0.0, 0.0, 0.0] # data coordinate where intersection occurs
        lineT = vtk.mutable(0.0) # parametric distance along line segment where intersection occurs
        pcoords = [0.0, 0.0, 0.0] # parametric location within cell (triangle) where intersection occurs
        subId = vtk.mutable(0) # sub id of cell intersection

        result = locator.IntersectWithLine(rayOrigin, rayEnd, tolerance, lineT, pt, pcoords, subId)

        return pt if result else None