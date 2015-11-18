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
from sensor import SensorObj
from controller import ControllerObj

class Simulator(object):

    def __init__(self):
        self.Sensor = SensorObj()
        self.Controller = ControllerObj(self.Sensor)
        self.Car = CarPlant(self.Controller)

    def mainLoop(self, endTime=10.0, dt=0.05):
        self.endTime = endTime
        self.t = np.arange(0.0, self.endTime, dt)
        self.stateOverTime = np.zeros((self.endTime/dt, 3))

        for idx, value in enumerate(self.t):
            self.stateOverTime[idx,:] = self.Car.simulateOneStep(value, dt)
            x = self.stateOverTime[idx,0] 
            y = self.stateOverTime[idx,1]
            theta = self.stateOverTime[idx,2] 
            self.setRobotState(x,y,theta)



    def run(self):

        self.timer = TimerCallback(targetFps=30)
        self.timer.callback = self.tick

        self.playTimer = TimerCallback(targetFps=30)
        self.playTimer.callback = self.tick3


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

        self.world = World.buildBigWorld()
        self.robot, self.frame = World.buildRobot()
        self.locator = World.buildCellLocator(self.world.polyData)

        self.Sensor.setLocator(self.locator)

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
        self.Car.setFrame(self.frame)
        self.mainLoop()

        applogic.resetCamera(viewDirection=[0.2,0,-1])
        view.showMaximized()
        view.raise_()

        app.start()


    def updateDrawIntersection(self, frame):

        origin = np.array(frame.transform.GetPosition())
        #print "origin is now at", origin
        d = DebugData()

        for i in xrange(self.Sensor.numRays):
            ray = self.Sensor.rays[:,i]
            rayTransformed = np.array(frame.transform.TransformNormal(ray))
            #print "rayTransformed is", rayTransformed
            intersection = self.Sensor.raycast(self.locator, origin, origin + rayTransformed*self.Sensor.rayLength)

            if intersection is not None:
                d.addLine(origin, intersection, color=[1,0,0])
            else:
                d.addLine(origin, origin+rayTransformed*self.Sensor.rayLength, color=[0,1,0])

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
        numStates = len(self.stateOverTime)

        idx = int(np.floor(numStates*p))

        x,y,theta = self.stateOverTime[idx]
        self.setRobotState(x,y,theta)

    def onSliderChanged(self, value):
        numSteps = len(self.stateOverTime)
        idx = int(np.floor(numSteps*(value/1000.0)))
        x,y,theta = self.stateOverTime[idx]
        self.setRobotState(x,y,theta)

    def onPlayButton(self):
        print 'play'
        self.playTimer.start()

        self.playTime = time.time()


if __name__ == "__main__":
    sim = Simulator()
    sim.run()

