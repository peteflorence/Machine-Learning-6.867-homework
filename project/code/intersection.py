import ddapp.vtkAll as vtk
import ddapp.visualization as vis
import ddapp.objectmodel as om
from ddapp.debugVis import DebugData
from ddapp.consoleapp import ConsoleApp
from ddapp import applogic
import numpy as np


def buildWorld():

    d = DebugData()
    d.addLine((2,-1,0), (2,1,0), radius=0.1)
    d.addLine((2,-1,0), (1,-2,0), radius=0.1)
    obj = vis.showPolyData(d.getPolyData(), 'world')
    return obj


def buildRobot():

    d = DebugData()
    d.addCone((0,0,0), (1,0,0), height=0.2, radius=0.1)
    obj = vis.showPolyData(d.getPolyData(), 'robot')
    frame = vis.addChildFrame(obj)
    return obj


def buildCellLocator(polyData):

    loc = vtk.vtkCellLocator()
    loc.SetDataSet(polyData)
    loc.BuildLocator()
    return loc


def computeIntersection(locator, rayOrigin, rayEnd):

    tolerance = 0.0 # intersection tolerance
    pt = [0.0, 0.0, 0.0] # data coordinate where intersection occurs
    lineT = vtk.mutable(0.0) # parametric distance along line segment where intersection occurs
    pcoords = [0.0, 0.0, 0.0] # parametric location within cell (triangle) where intersection occurs
    subId = vtk.mutable(0) # sub id of cell intersection

    result = locator.IntersectWithLine(rayOrigin, rayEnd, tolerance, lineT, pt, pcoords, subId)

    return pt if result else None


def updateIntersection(frame):

    origin = np.array(frame.transform.GetPosition())
    rayLength = 5

    for i in range(0,numRays):
        ray = rays[:,i]
        rayTransformed = np.array(frame.transform.TransformNormal(ray))
        intersection = computeIntersection(locator, origin, origin + rayTransformed*rayLength)
        name = 'ray intersection ' + str(i)

        if intersection is not None:
            om.removeFromObjectModel(om.findObjectByName(name))
            d = DebugData()
            d.addLine(origin, intersection)
            color = [1,0,0]
            # d.addSphere(intersection, radius=0.04)
            vis.updatePolyData(d.getPolyData(), name, color=color)
        else:
            om.removeFromObjectModel(om.findObjectByName(name))
            d = DebugData()
            d.addLine(origin, origin+rayTransformed*rayLength)
            color = [0,1,0]
            # d.addSphere(intersection, radius=0.04)
            vis.updatePolyData(d.getPolyData(), name, color=color)


#########################
numRays = 20
angleMin = -np.pi/2
angleMax = np.pi/2
angleGrid = np.linspace(angleMin, angleMax, numRays)
rays = np.zeros((3,numRays))
rays[0,:] = np.cos(angleGrid)
rays[1,:] = np.sin(angleGrid)



app = ConsoleApp()
view = app.createView()

world = buildWorld()
robot = buildRobot()
locator = buildCellLocator(world.polyData)

frame = robot.getChildFrame()
frame.setProperty('Edit', True)
frame.widget.HandleRotationEnabledOff()
rep = frame.widget.GetRepresentation()
rep.SetTranslateAxisEnabled(2, False)
rep.SetRotateAxisEnabled(0, False)
rep.SetRotateAxisEnabled(1, False)

frame.connectFrameModified(updateIntersection)
updateIntersection(frame)

applogic.resetCamera(viewDirection=[0.2,0,-1])
view.showMaximized()
view.raise_()
app.start()
