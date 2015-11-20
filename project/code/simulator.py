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
import argparse

from PythonQt import QtCore, QtGui

from world import World
from car import CarPlant
from sensor import SensorObj
from controller import ControllerObj
from sarsa import SARSA
from reward import Reward


class Simulator(object):

    def __init__(self, numObstacles=200, endTime=40):
        self.startSimTime = time.time()
        self.Sensor = SensorObj()
        self.Controller = ControllerObj(self.Sensor)
        self.Car = CarPlant(self.Controller)
        self.Sarsa = SARSA(sensorObj=self.Sensor, actionSet=self.Controller.actionSet)
        self.collisionThreshold = 1.3
        self.Reward = Reward(self.Sensor, collisionThreshold=self.collisionThreshold)
        self.numObstacles = numObstacles
        self.endTime = endTime
        self.initialize()

    def initialize(self):

        # create the visualizer object
        self.app = ConsoleApp()
        # view = app.createView(useGrid=False)
        self.view = self.app.createView(useGrid=False)

        # panel = QtGui.QWidget()
        # l = QtGui.QHBoxLayout(panel)
        #
        # playButton = QtGui.QPushButton('Play')
        # playButton.connect('clicked()', self.onPlayButton)
        #
        # slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        # slider.connect('valueChanged(int)', self.onSliderChanged)
        # slider.setMaximum(999)
        #
        # l.addWidget(playButton)
        # l.addWidget(slider)
        #
        # w = QtGui.QWidget()
        # l = QtGui.QVBoxLayout(w)
        # l.addWidget(self.view)
        # l.addWidget(panel)



        # create the things needed for simulation
        self.world = World.buildBigWorld(numObstacles=self.numObstacles)
        self.robot, self.frame = World.buildRobot()
        self.locator = World.buildCellLocator(self.world.polyData)
        self.Sensor.setLocator(self.locator)
        self.frame = self.robot.getChildFrame()


        self.frame.setProperty('Scale', 3)
        self.frame.setProperty('Edit', True)
        self.frame.widget.HandleRotationEnabledOff()
        rep = self.frame.widget.GetRepresentation()
        rep.SetTranslateAxisEnabled(2, False)
        rep.SetRotateAxisEnabled(0, False)
        rep.SetRotateAxisEnabled(1, False)

        # Simulate
        self.Car.setFrame(self.frame)
        # self.mainLoop()


    def mainLoop(self, endTime=None, dt=0.05):

        if endTime is not None:
            self.endTime = endTime
        self.t = np.arange(0.0, self.endTime, dt)
        numTimesteps = np.size(self.t)
        self.stateOverTime = np.zeros((numTimesteps, 3))
        self.raycastData = np.zeros((numTimesteps, self.Sensor.numRays))
        self.controlInputData = np.zeros(numTimesteps)
        self.rewardData = np.zeros(numTimesteps)

        currentCarState = np.copy(self.Car.state)
        nextCarState = np.copy(self.Car.state)
        self.setRobotState(currentCarState[0], currentCarState[1], currentCarState[2])
        currentRaycast = self.Sensor.raycastAll(self.frame)
        nextRaycast = np.zeros(self.Sensor.numRays)
        counter = 0


        for idx in xrange(np.size(self.t) - 1):
            currentTime = self.t[idx]
            self.stateOverTime[idx,:] = currentCarState
            x = self.stateOverTime[idx,0]
            y = self.stateOverTime[idx,1]
            theta = self.stateOverTime[idx,2]
            self.setRobotState(x,y,theta)
            # self.setRobotState(currentCarState[0], currentCarState[1], currentCarState[2])
            currentRaycast = self.Sensor.raycastAll(self.frame)
            self.raycastData[idx,:] = currentRaycast
            controlInput, controlInputIdx = self.Controller.computeControlInput(currentCarState, currentTime, self.frame, raycastDistance=currentRaycast)
            self.controlInputData[idx] = controlInput

            nextCarState = self.Car.simulateOneStep(controlInput=controlInput)

            # want to compute nextRaycast so we can do the SARSA algorithm
            x = nextCarState[0]
            y = nextCarState[1]
            theta = nextCarState[2]
            self.setRobotState(x,y,theta)
            nextRaycast = self.Sensor.raycastAll(self.frame)

            # compute the next control input, this is needed for SARSA
            nextControlInput, nextControlInputIdx = self.Controller.computeControlInput(nextCarState, currentTime, self.frame,
                                                                                        raycastDistance=nextRaycast)

            # Gather all the information we have
            S_current = (currentCarState, currentRaycast)
            S_next = (nextCarState, nextRaycast)


            # compute the reward
            reward = self.Reward.computeReward(S_next, controlInput)
            self.rewardData[idx] = reward

            ## SARSA update
            self.Sarsa.sarsaUpdate(S_current, controlInputIdx, reward, S_next, nextControlInputIdx)

            #bookkeeping
            currentCarState = nextCarState
            currentRaycast = nextRaycast
            counter+=1
            # break if we are in collision
            if self.checkInCollision(nextRaycast):
                print "Had a collision, terminating simulation"
                break

        # fill in the last state by hand
        self.stateOverTime[counter,:] = currentCarState
        self.raycastData[counter,:] = currentRaycast

        # truncate stateOverTime, raycastData, controlInputs to be the correct size
        self.stateOverTime = self.stateOverTime[0:counter+1, :]
        self.raycastData = self.raycastData[0:counter+1, :]
        self.controlInputData = self.controlInputData[0:counter+1]
        self.rewardData = self.rewardData[0:counter+1]

        self.counter = counter
        self.endTime = 1.0*counter/numTimesteps*self.endTime


    def runOld(self):

        self.timer = TimerCallback(targetFps=30)
        self.timer.callback = self.tick

        self.playTimer = TimerCallback(targetFps=30)
        self.playTimer.callback = self.tick3


        app = ConsoleApp()
        view = app.createView(useGrid=False)
        self.view = view

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

        self.frame.setProperty('Scale', 3)
        self.frame.setProperty('Edit', True)
        self.frame.widget.HandleRotationEnabledOff()
        rep = self.frame.widget.GetRepresentation()
        rep.SetTranslateAxisEnabled(2, False)
        rep.SetRotateAxisEnabled(0, False)
        rep.SetRotateAxisEnabled(1, False)

        # Simulate
        self.Car.setFrame(self.frame)
        self.mainLoop()

        self.frame.connectFrameModified(self.updateDrawIntersection)
        self.updateDrawIntersection(self.frame)

        applogic.resetCamera(viewDirection=[0.2,0,-1])
        view.showMaximized()
        view.raise_()

        elapsed = time.time() - self.startSimTime
        simRate = self.counter/elapsed
        print "Total run time", elapsed
        print "Ticks (Hz)", simRate
        app.start()


    def setupPlayback(self):

        self.timer = TimerCallback(targetFps=30)
        self.timer.callback = self.tick

        self.playTimer = TimerCallback(targetFps=30)
        self.playTimer.callback = self.tick3

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
        l.addWidget(self.view)
        l.addWidget(panel)
        w.showMaximized()



        self.frame.connectFrameModified(self.updateDrawIntersection)
        self.updateDrawIntersection(self.frame)

        applogic.resetCamera(viewDirection=[0.2,0,-1])
        self.view.showMaximized()
        self.view.raise_()

        elapsed = time.time() - self.startSimTime
        simRate = self.counter/elapsed
        print "Total run time", elapsed
        print "Ticks (Hz)", simRate
        print "Number of steps taken", self.counter
        self.app.start()

    def run(self):
        self.initialize()
        self.mainLoop()
        self.setupPlayback()

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

        #camera = self.view.camera()
        #camera.SetFocalPoint(frame.transform.GetPosition())
        #camera.SetPosition(frame.transform.TransformPoint((-30,0,10)))


    def setRobotState(self, x, y, theta):
        t = vtk.vtkTransform()
        t.Translate(x,y,0.0)
        t.RotateZ(np.degrees(theta))
        self.robot.getChildFrame().copyFrame(t)

    # returns true if we are in collision
    def checkInCollision(self, raycastDistance):
        if np.min(raycastDistance) < self.collisionThreshold:
            return True
        else:
            return False

    def tick(self):
        #print timer.elapsed
        #simulate(t.elapsed)
        x = np.sin(time.time())
        y = np.cos(time.time())
        self.setRobotState(x,y,0.0)
        if (time.time() - self.playTime) > self.endTime:
            self.playTimer.stop()

    def tick2(self):
        newtime = time.time() - self.playTime
        print time.time() - self.playTime
        x = np.sin(newtime)
        y = np.cos(newtime)
        self.setRobotState(x,y,0.0)

    def tick3(self):
        newtime = time.time() - self.playTime
        if newtime > self.endTime:
            self.playTimer.stop()
        newtime = np.clip(newtime, 0, self.endTime)

        p = newtime/self.endTime
        numStates = len(self.stateOverTime)

        idx = int(np.floor(numStates*p))
        idx = min(idx, numStates-1)
        x,y,theta = self.stateOverTime[idx]
        self.setRobotState(x,y,theta)

    def onSliderChanged(self, value):
        self.playTimer.stop()
        numSteps = len(self.stateOverTime)
        idx = int(np.floor(numSteps*(value/1000.0)))
        idx = min(idx, numSteps-1)
        x,y,theta = self.stateOverTime[idx]
        self.setRobotState(x,y,theta)

    def onPlayButton(self):
        print 'play'
        self.playTimer.start()
        self.playTime = time.time()




def main(argv):
    sim = Simulator(numObstacles=200)
    sim.run()


if __name__ == "__main__":
    # main(sys.argv[1:])
    parser = argparse.ArgumentParser(description='interpret simulation parameters')
    parser.add_argument('--numObstacles', type=int, nargs=1, default=[100])
    parser.add_argument('--endTime', type=int, nargs=1, default=[40])
    argNamespace = parser.parse_args()
    numObstacles = argNamespace.numObstacles[0]
    endTime = argNamespace.endTime[0]
    sim = Simulator(numObstacles=numObstacles, endTime=endTime)
    sim.run()


