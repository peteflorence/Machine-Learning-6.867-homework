import ddapp.vtkAll as vtk
import ddapp.visualization as vis
import ddapp.objectmodel as om
from ddapp.debugVis import DebugData
from ddapp.consoleapp import ConsoleApp
from ddapp.timercallback import TimerCallback
from ddapp import applogic
import numpy as np
import time


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


def updateDrawIntersection(frame):

    origin = np.array(frame.transform.GetPosition())
    #print "origin is now at", origin
    
    for i in xrange(numRays):
        ray = rays[:,i]
        rayTransformed = np.array(frame.transform.TransformNormal(ray))
        #print "rayTransformed is", rayTransformed
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



# initial positions
x = 0.0
y = 0.0
psi = 0.0

rad = np.pi/180.0

# initial state
state = np.array([x, y, psi*rad])

def simulate(dt):
    
    t = np.arange(0.0, 10, dt)
    y = integrate.odeint(dynamics, state, t)


def computeRays(frame):

    intersections = np.zeros((3,numRays))

    origin = np.array(frame.transform.GetPosition())

    for i in xrange(0,numRays):
        ray = rays[:,i]
        rayTransformed = np.array(frame.transform.TransformNormal(ray))
        intersections[i] = computeIntersection(locator, origin, origin + rayTransformed*rayLength)

    return intersections

def calcInput(state, t):

    u = 0
    intersections = computeRays(frame)
    return u

#     #Barry 12 controller
#     c_1 = 1
#     c_2 = 10
#     c_3 = 100

#     F = rays*0.0
#     for i in range(0,numRays):
#         F[i] = -c_1*

def dynamics(state, t, u):

    dqdt = np.zeros_like(state)
    
    dqdt[0] = v*np.cos(state[2])
    dqdt[1] = v*np.sin(state[2]) 
    dqdt[2] = calcInput(state)
    
    return dqdt



def setRobotState(x,y,theta):
    t = vtk.vtkTransform()
    t.Translate(x,y,0.0)
    t.RotateZ(np.degrees(theta))
    robot.getChildFrame().copyFrame(t)


def tick():
    #print timer.elapsed
    #simulate(t.elapsed)

    x = np.sin(time.time())
    y = np.cos(time.time())
    setRobotState(x,y,0.0)


#########################
numRays = 20
rayLength = 5
angleMin = -np.pi/2
angleMax = np.pi/2
angleGrid = np.linspace(angleMin, angleMax, numRays)
rays = np.zeros((3,numRays))
rays[0,:] = np.cos(angleGrid)
rays[1,:] = np.sin(angleGrid)


timer = TimerCallback(targetFps=30)
timer.callback = tick




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

frame.connectFrameModified(updateDrawIntersection)
updateDrawIntersection(frame)

applogic.resetCamera(viewDirection=[0.2,0,-1])
view.showMaximized()
view.raise_()
app.start()
