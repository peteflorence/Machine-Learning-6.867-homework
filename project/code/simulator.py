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

from world import World
from car import CarPlant

class Simulator(object):

    def __init__(self):
        self.variable = 1

    def run(self):

        Car = CarPlant()


        #########################
        self.numRays = 20
        self.rayLength = 5
        self.angleMin = -np.pi/2
        self.angleMax = np.pi/2
        self.angleGrid = np.linspace(self.angleMin, self.angleMax, self.numRays)
        self.rays = np.zeros((3,self.numRays))
        self.rays[0,:] = np.cos(self.angleGrid)
        self.rays[1,:] = np.sin(self.angleGrid)


        self.timer = TimerCallback(targetFps=30)
        self.timer.callback = self.tick

        self.playTimer = TimerCallback(targetFps=30)
        self.playTimer.callback = self.tick3
        self.endTime = 10


        app = ConsoleApp()
        view = app.createView()

        panel = QtGui.QWidget()
        l = QtGui.QHBoxLayout(panel)

        playButton = QtGui.QPushButton('Play')
        playButton.connect('clicked()', self.onPlayButton)

        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        slider.connect('valueChanged(int)', self.onSliderChanged)
        slider.setMaximum(999)

        l.addWidget(playButton)
        l.addWidget(slider)

        w = QtGui.QWidget()
        l = QtGui.QVBoxLayout(w)
        l.addWidget(view)
        l.addWidget(panel)

        w.showMaximized()

        self.world = World.buildSimpleWorld()
        self.robot, self.frame = World.buildRobot()
        self.locator = World.buildCellLocator(self.world.polyData)

        self.frame = self.robot.getChildFrame()
        self.frame.setProperty('Edit', True)
        self.frame.widget.HandleRotationEnabledOff()
        rep = self.frame.widget.GetRepresentation()
        rep.SetTranslateAxisEnabled(2, False)
        rep.SetRotateAxisEnabled(0, False)
        rep.SetRotateAxisEnabled(1, False)

        self.frame.connectFrameModified(self.updateDrawIntersection)
        self.updateDrawIntersection(self.frame)

        # Simulate
        self.evolvedState = Car.simulate()

        applogic.resetCamera(viewDirection=[0.2,0,-1])
        view.showMaximized()
        view.raise_()
        app.start()


    def computeRays(self,frame):

        intersections = np.zeros((3,self.numRays))

        origin = np.array(frame.transform.GetPosition())

        for i in range(0,self.numRays):
            ray = self.rays[:,i]
            rayTransformed = np.array(frame.transform.TransformNormal(ray))
            intersections[:,i] = self.computeIntersection(self.locator, origin, origin + rayTransformed*self.rayLength)

        return intersections

    def calcInput(self, state, t):

        #u = 0
        u = np.sin(t)
        intersections = self.computeRays(self.frame)
        return u

    #     #Barry 12 controller
    #     c_1 = 1
    #     c_2 = 10
    #     c_3 = 100

    #     F = rays*0.0
    #     for i in range(0,numRays):
    #         F[i] = -c_1*


    def computeIntersection(self, locator, rayOrigin, rayEnd):

        tolerance = 0.0 # intersection tolerance
        pt = [0.0, 0.0, 0.0] # data coordinate where intersection occurs
        lineT = vtk.mutable(0.0) # parametric distance along line segment where intersection occurs
        pcoords = [0.0, 0.0, 0.0] # parametric location within cell (triangle) where intersection occurs
        subId = vtk.mutable(0) # sub id of cell intersection

        result = locator.IntersectWithLine(rayOrigin, rayEnd, tolerance, lineT, pt, pcoords, subId)

        return pt if result else None


    def updateDrawIntersection(self, frame):

        origin = np.array(frame.transform.GetPosition())
        #print "origin is now at", origin
        d = DebugData()

        for i in xrange(self.numRays):
            ray = self.rays[:,i]
            rayTransformed = np.array(frame.transform.TransformNormal(ray))
            #print "rayTransformed is", rayTransformed
            intersection = self.computeIntersection(self.locator, origin, origin + rayTransformed*self.rayLength)


            if intersection is not None:
                d.addLine(origin, intersection, color=[1,0,0])
            else:
                d.addLine(origin, origin+rayTransformed*self.rayLength, color=[0,1,0])

        vis.updatePolyData(d.getPolyData(), 'rays', colorByName='RGB255')


    def setRobotState(self, x, y, theta):
        t = vtk.vtkTransform()
        t.Translate(x,y,0.0)
        t.RotateZ(np.degrees(theta))
        self.robot.getChildFrame().copyFrame(t)


    def tick(self):
        #print timer.elapsed
        #simulate(t.elapsed)
        x = np.sin(time.time())
        y = np.cos(time.time())
        self.setRobotState(x,y,0.0)

    def tick2(self):
        newtime = time.time() - self.playTime
        print time.time() - self.playTime
        x = np.sin(newtime)
        y = np.cos(newtime)
        self.setRobotState(x,y,0.0)

    def tick3(self):
        newtime = time.time() - self.playTime
        newtime = np.clip(newtime, 0, self.endTime)

        p = newtime/self.endTime
        numStates = len(self.evolvedState)

        idx = int(np.floor(numStates*p))

        x,y,theta = self.evolvedState[idx]
        self.setRobotState(x,y,theta)

    def onSliderChanged(self, value):
        numSteps = len(self.evolvedState)
        idx = int(np.floor(numSteps*(value/1000.0)))
        print value, idx
        x,y,theta = self.evolvedState[idx]
        self.setRobotState(x,y,theta)

    def onPlayButton(self):
        print 'play'
        self.playTimer.start()

        self.playTime = time.time()


if __name__ == "__main__":
    sim = Simulator()
    sim.run()

