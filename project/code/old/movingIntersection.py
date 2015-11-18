import ddapp.vtkAll as vtk
import ddapp.visualization as vis
import ddapp.objectmodel as om
from ddapp.debugVis import DebugData
from ddapp.consoleapp import ConsoleApp
from ddapp.timercallback import TimerCallback
from ddapp import applogic
import numpy as np
import time
import scipy.integrate as integrate

from PythonQt import QtCore, QtGui

class Car:

    def __init__(self):
        self.x = 0





# initial positions
x = 0.0
y = 0.0
psi = 0.0
rad = np.pi/180.0

# initial state
state = np.array([x, y, psi*rad])

# constant velocity
v = 8



def computeRays(frame):

    intersections = np.zeros((3,numRays))

    origin = np.array(frame.transform.GetPosition())

    for i in range(0,numRays):
        ray = rays[:,i]
        rayTransformed = np.array(frame.transform.TransformNormal(ray))
        intersections[:,i] = computeIntersection(locator, origin, origin + rayTransformed*rayLength)

    return intersections

def calcInput(state, t):

    #u = 0
    u = np.sin(t)
    intersections = computeRays(frame)
    return u

#     #Barry 12 controller
#     c_1 = 1
#     c_2 = 10
#     c_3 = 100

#     F = rays*0.0
#     for i in range(0,numRays):
#         F[i] = -c_1*

def dynamics(state, t):

    dqdt = np.zeros_like(state)
    
    u = calcInput(state, t)

    dqdt[0] = v*np.cos(state[2])
    dqdt[1] = v*np.sin(state[2]) 
    dqdt[2] = u
    
    return dqdt

endTime = 10
def simulate(dt=0.05):
    
    t = np.arange(0.0, endTime, dt)
    y = integrate.odeint(dynamics, state, t)
    print y
    return y


#output = simulate(0.05)



def buildWorld():

    d = DebugData()
    d.addLine((2,-1,0), (2,1,0), radius=0.1)
    d.addLine((2,-1,0), (1,-2,0), radius=0.1)
    obj = vis.showPolyData(d.getPolyData(), 'world')
    return obj


def buildRobot(x=0,y=0):

    d = DebugData()
    d.addCone((x,y,0), (1,0,0), height=0.2, radius=0.1)
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
    d = DebugData()

    for i in xrange(numRays):
        ray = rays[:,i]
        rayTransformed = np.array(frame.transform.TransformNormal(ray))
        #print "rayTransformed is", rayTransformed
        intersection = computeIntersection(locator, origin, origin + rayTransformed*rayLength)


        if intersection is not None:
            d.addLine(origin, intersection, color=[1,0,0])
        else:
            d.addLine(origin, origin+rayTransformed*rayLength, color=[0,1,0])

    vis.updatePolyData(d.getPolyData(), 'rays', colorByName='RGB255')



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

def tick2():
    
    newtime = time.time() - playTime
    print time.time() - playTime

    x = np.sin(newtime)
    y = np.cos(newtime)

    setRobotState(x,y,0.0)

def tick3():

    newtime = time.time() - playTime
    
    newtime = np.clip(newtime, 0, endTime)

    p = newtime/endTime
    numStates = len(evolvedState)

    idx = int(np.floor(numStates*p))

    x,y,theta = evolvedState[idx]
    setRobotState(x,y,theta)


def onSliderChanged(value):
    numSteps = len(evolvedState)
    idx = int(np.floor(numSteps*(value/1000.0)))
    print value, idx

    x,y,theta = evolvedState[idx]
    setRobotState(x,y,theta)


def onPlayButton():
    print 'play'
    playTimer.start()

    global playTime 
    playTime = time.time()




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


playTimer = TimerCallback(targetFps=30)
playTimer.callback = tick3






app = ConsoleApp()
view = app.createView()


panel = QtGui.QWidget()
l = QtGui.QHBoxLayout(panel)

playButton = QtGui.QPushButton('Play')
playButton.connect('clicked()', onPlayButton)


slider = QtGui.QSlider(QtCore.Qt.Horizontal)
slider.connect('valueChanged(int)', onSliderChanged)
slider.setMaximum(999)

l.addWidget(playButton)
l.addWidget(slider)


w = QtGui.QWidget()
l = QtGui.QVBoxLayout(w)
l.addWidget(view)
l.addWidget(panel)


w.showMaximized()

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

# Simulate
evolvedState = simulate()

applogic.resetCamera(viewDirection=[0.2,0,-1])
view.showMaximized()
view.raise_()
app.start()
