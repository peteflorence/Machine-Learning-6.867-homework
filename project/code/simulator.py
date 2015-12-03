import ddapp.vtkAll as vtk
import ddapp.visualization as vis
import ddapp.objectmodel as om
from ddapp.debugVis import DebugData
from ddapp.consoleapp import ConsoleApp
from ddapp.timercallback import TimerCallback
from ddapp import applogic
from ddapp import transformUtils
import numpy as np
import time
import scipy.integrate as integrate
import argparse
import matplotlib.pyplot as plt

from PythonQt import QtCore, QtGui

from world import World
from car import CarPlant
from sensor import SensorObj
from controller import ControllerObj
from sarsaContinuous import SARSAContinuous
from sarsaDiscrete import SARSADiscrete
from reward import Reward


class Simulator(object):


    def __init__(self, percentObsDensity=20, endTime=40, randomizeControl=False, nonRandomWorld=False,
                 circleRadius=0.7, worldScale=1.0, supervisedTrainingTime=500, autoInitialize=True, verbose=True,
                 sarsaType="discrete"):
        self.verbose = verbose
        self.randomizeControl = randomizeControl
        self.startSimTime = time.time()
        self.collisionThreshold = 1.3
        self.randomSeed = 5
        self.Sarsa_numInnerBins = 4
        self.Sarsa_numOuterBins = 4
        self.Sensor_rayLength = 8
        self.sarsaType = sarsaType

        self.percentObsDensity = percentObsDensity
        self.supervisedTrainingTime = 3000
        self.learningTime = 9000
        self.defaultControllerTime = 1000
        self.nonRandomWorld = nonRandomWorld
        self.circleRadius = circleRadius
        self.worldScale = worldScale
        self.supervisedTrainingTime = supervisedTrainingTime
        # create the visualizer object
        self.app = ConsoleApp()
        # view = app.createView(useGrid=False)
        self.view = self.app.createView(useGrid=False)
        if autoInitialize:
            self.initialize()

    def initialize(self):

        self.endTime = self.supervisedTrainingTime + self.learningTime + self.defaultControllerTime
        self.Sensor = SensorObj(rayLength=self.Sensor_rayLength)
        self.Controller = ControllerObj(self.Sensor)
        self.Car = CarPlant(self.Controller)
        self.Reward = Reward(self.Sensor, collisionThreshold=self.collisionThreshold)
        self.setSARSA()

        # create the things needed for simulation
        self.world = World.buildCircleWorld(percentObsDensity=self.percentObsDensity, circleRadius=self.circleRadius,
                                            nonRandom=self.nonRandomWorld, scale=self.worldScale,
                                            randomSeed=self.randomSeed)
        self.robot, self.frame = World.buildRobot()
        self.locator = World.buildCellLocator(self.world.visObj.polyData)
        self.Sensor.setLocator(self.locator)
        self.frame = self.robot.getChildFrame()
        self.frame.setProperty('Scale', 3)
        self.frame.setProperty('Edit', True)
        self.frame.widget.HandleRotationEnabledOff()
        rep = self.frame.widget.GetRepresentation()
        rep.SetTranslateAxisEnabled(2, False)
        rep.SetRotateAxisEnabled(0, False)
        rep.SetRotateAxisEnabled(1, False)

        self.Car.setFrame(self.frame)
        print "Finished initialization"

    def setSARSA(self, type=None):
        if type is None:
            type = self.sarsaType

        if type=="discrete":
            self.Sarsa = SARSADiscrete(sensorObj=self.Sensor, actionSet=self.Controller.actionSet,
                                            collisionThreshold=self.collisionThreshold,
                                   numInnerBins = self.Sarsa_numInnerBins, numOuterBins = self.Sarsa_numOuterBins)
        elif type=="continuous":
            self.Sarsa = SARSAContinuous(sensorObj=self.Sensor, actionSet=self.Controller.actionSet,
                                        collisionThreshold=self.collisionThreshold)
        else:
            raise ValueError("sarsa type must be either discrete or continuous")


    def runSingleSimulation(self, useQValueController=False, randomizeDefaultController=True, updateQValues=True):

        if self.verbose: print "using QValue based controller = ", useQValueController

        self.setCollisionFreeInitialState()

        currentCarState = np.copy(self.Car.state)
        nextCarState = np.copy(self.Car.state)
        self.setRobotFrameState(currentCarState[0], currentCarState[1], currentCarState[2])
        currentRaycast = self.Sensor.raycastAll(self.frame)
        nextRaycast = np.zeros(self.Sensor.numRays)
        self.Sarsa.resetElibilityTraces()

        # record the reward data
        runData = dict()
        discountedReward = 0
        avgReward = 0
        startIdx = self.counter


        while (self.counter < self.numTimesteps - 1):
            idx = self.counter
            currentTime = self.t[idx]
            self.stateOverTime[idx,:] = currentCarState
            x = self.stateOverTime[idx,0]
            y = self.stateOverTime[idx,1]
            theta = self.stateOverTime[idx,2]
            self.setRobotFrameState(x,y,theta)
            # self.setRobotState(currentCarState[0], currentCarState[1], currentCarState[2])
            currentRaycast = self.Sensor.raycastAll(self.frame)
            self.raycastData[idx,:] = currentRaycast
            S_current = (currentCarState, currentRaycast)

            if useQValueController:
                counterForGreedyDecay = self.counter - self.usingQValueControllerIdx
                controlInput, controlInputIdx, emptyQValue = self.Sarsa.computeGreedyControlPolicy(S_current,
                                                                                                   counter=counterForGreedyDecay)
                if emptyQValue:
                    self.emptyQValue[idx] = 1


            if not useQValueController or emptyQValue:
                controlInput, controlInputIdx = self.Controller.computeControlInput(currentCarState,
                                                                                currentTime, self.frame,
                                                                                raycastDistance=currentRaycast,
                                                                                randomize=randomizeDefaultController)
            self.controlInputData[idx] = controlInput

            nextCarState = self.Car.simulateOneStep(controlInput=controlInput)

            # want to compute nextRaycast so we can do the SARSA algorithm
            x = nextCarState[0]
            y = nextCarState[1]
            theta = nextCarState[2]
            self.setRobotFrameState(x,y,theta)
            nextRaycast = self.Sensor.raycastAll(self.frame)


            # Gather all the information we have
            S_next = (nextCarState, nextRaycast)

            if useQValueController:
                nextControlInput, nextControlInputIdx, emptyQValue = self.Sarsa.computeGreedyControlPolicy(S_next)

            if not useQValueController or emptyQValue:
                nextControlInput, nextControlInputIdx = self.Controller.computeControlInput(nextCarState, currentTime, self.frame,
                                                                                        raycastDistance=nextRaycast)


            # compute the next control input, this is needed for SARSA



            # compute the reward
            reward = self.Reward.computeReward(S_next, controlInput)
            self.rewardData[idx] = reward

            discountedReward += self.Sarsa.gamma**(self.counter-startIdx)*reward
            avgReward += reward

            ## SARSA update
            if updateQValues:
                self.Sarsa.sarsaUpdate(S_current, controlInputIdx, reward, S_next, nextControlInputIdx)

            #bookkeeping
            currentCarState = nextCarState
            currentRaycast = nextRaycast
            self.counter+=1
            # break if we are in collision
            if self.checkInCollision(nextRaycast):
                if self.verbose: print "Had a collision, terminating simulation"
                break


        # fill in the last state by hand
        self.stateOverTime[self.counter,:] = currentCarState
        self.raycastData[self.counter,:] = currentRaycast

        # return the total reward
        avgReward = avgReward*1.0/max(1, self.counter -startIdx)
        # this extra multiplication is so that it is in the same "units" as avgReward
        runData['discountedReward'] = discountedReward*(1 - self.Sarsa.gamma)
        runData['avgReward'] = avgReward


        # this just makes sure we don't get stuck in an infinite loop.
        if startIdx == self.counter:
            self.counter += 1

        return runData


    def runBatchSimulation(self, endTime=None, dt=0.05):

        # for use in playback
        self.dt = dt

        self.endTime = self.supervisedTrainingTime + self.learningTime + self.defaultControllerTime

        self.t = np.arange(0.0, self.endTime, dt)
        numTimesteps = np.size(self.t)
        self.stateOverTime = np.zeros((numTimesteps, 3))
        self.raycastData = np.zeros((numTimesteps, self.Sensor.numRays))
        self.controlInputData = np.zeros(numTimesteps)
        self.rewardData = np.zeros(numTimesteps)
        self.numTimesteps = numTimesteps
        self.emptyQValue = np.zeros(numTimesteps)

        self.counter = 0
        self.simulationData = []
    
        self.initializeStatusBar()


        # three while loops for different phases of simulation, supervisedTraining, learning, default
        while (self.counter < self.supervisedTrainingTime/dt):
            self.printStatusBar()
            useQValueController = False
            runData = dict()
            runData['startIdx'] = self.counter
            rd = self.runSingleSimulation(useQValueController=useQValueController, randomizeDefaultController=True,
                                     updateQValues=True)
            runData['avgReward'] = rd['avgReward']
            runData['discountedReward'] = rd['discountedReward']
            runData['controllerType'] = "defaultRandom"
            runData['duration'] = self.counter - runData['startIdx']
            runData['endIdx'] = self.counter
            self.simulationData.append(runData)


        self.usingQValueControllerIdx = self.counter
        while (self.counter < (self.supervisedTrainingTime + self.learningTime)/dt):
            self.printStatusBar()
            runData = dict()
            runData['startIdx'] = self.counter
            useQValueController = True
            rd = self.runSingleSimulation(useQValueController=useQValueController, randomizeDefaultController=True,
                                     updateQValues=True)

            runData['avgReward'] = rd['avgReward']
            runData['discountedReward'] = rd['discountedReward']
            runData['controllerType'] = "QValue"
            runData['duration'] = self.counter - runData['startIdx']
            runData['endIdx'] = self.counter
            self.simulationData.append(runData)


        self.usingDefaultControllerIdx = self.counter
        while (self.counter < self.numTimesteps - 1):
            self.printStatusBar()
            runData = dict()
            runData['startIdx'] = self.counter
            useQValueController = False
            rd = self.runSingleSimulation(useQValueController=useQValueController, randomizeDefaultController=False,
                                     updateQValues=False)

            runData['avgReward'] = rd['avgReward']
            runData['discountedReward'] = rd['discountedReward']
            runData['controllerType'] = "default"
            runData['duration'] = self.counter - runData['startIdx']
            runData['endIdx'] = self.counter
            self.simulationData.append(runData)


        # BOOKKEEPING
        # truncate stateOverTime, raycastData, controlInputs to be the correct size
        self.stateOverTime = self.stateOverTime[0:self.counter+1, :]
        self.raycastData = self.raycastData[0:self.counter+1, :]
        self.controlInputData = self.controlInputData[0:self.counter+1]
        self.rewardData = self.rewardData[0:self.counter+1]
        self.endTime = 1.0*self.counter/numTimesteps*self.endTime


    def initializeStatusBar(self):
        self.numTicks = 10
        self.nextTickComplete = 1.0 / float(self.numTicks)
        self.nextTickIdx = 1
        print "Simulation percentage complete: (", self.numTicks, " # is complete)"
    
    def printStatusBar(self):
        fractionDone = float(self.counter) / float(self.numTimesteps)
        if fractionDone > self.nextTickComplete:

            self.nextTickIdx += 1
            self.nextTickComplete += 1.0 / self.numTicks

            timeSoFar = time.time() - self.startSimTime 
            estimatedTimeLeft_sec = (1 - fractionDone) * timeSoFar / fractionDone
            estimatedTimeLeft_minutes = estimatedTimeLeft_sec / 60.0

            print "#" * self.nextTickIdx, "-" * (self.numTicks - self.nextTickIdx), "estimated", estimatedTimeLeft_minutes, "minutes left"



    def setCollisionFreeInitialState(self):
        tol = 5

        while True:
            x = np.random.uniform(self.world.Xmin+tol, self.world.Xmax-tol, 1)[0]
            y = np.random.uniform(self.world.Ymin+tol, self.world.Ymax-tol, 1)[0]
            theta = np.random.uniform(0,2*np.pi,1)[0]
            self.Car.setCarState(x,y,theta)
            self.setRobotFrameState(x,y,theta)

            if not self.checkInCollision():
                break

        # if self.checkInCollision():
        #     print "IN COLLISION"
        # else:
        #     print "COLLISION FREE"

        return x,y,theta

    def setupPlayback(self):

        self.timer = TimerCallback(targetFps=30)
        self.timer.callback = self.tick

        playButtonFps = 1.0/self.dt
        print "playButtonFPS", playButtonFps
        self.playTimer = TimerCallback(targetFps=playButtonFps)
        self.playTimer.callback = self.playTimerCallback
        self.sliderMovedByPlayTimer = False

        panel = QtGui.QWidget()
        l = QtGui.QHBoxLayout(panel)

        playButton = QtGui.QPushButton('Play/Pause')
        playButton.connect('clicked()', self.onPlayButton)

        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        slider.connect('valueChanged(int)', self.onSliderChanged)
        self.sliderMax = self.numTimesteps
        slider.setMaximum(self.sliderMax)
        self.slider = slider

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
        self.counter = 1
        self.runBatchSimulation()
        # self.Sarsa.plotWeights()
        self.setupPlayback()

    def updateDrawIntersection(self, frame):

        origin = np.array(frame.transform.GetPosition())
        #print "origin is now at", origin
        d = DebugData()

        sliderIdx = self.slider.value

        if (sliderIdx >= self.usingQValueControllerIdx) and (sliderIdx < self.usingDefaultControllerIdx) \
                and not self.emptyQValue[sliderIdx]:
            colorMaxRange = [1,1,0]
        elif sliderIdx >= self.usingDefaultControllerIdx:
            colorMaxRange = [0,1,0]
        else:
            colorMaxRange = [0,0,1]

        for i in xrange(self.Sensor.numRays):
            ray = self.Sensor.rays[:,i]
            rayTransformed = np.array(frame.transform.TransformNormal(ray))
            #print "rayTransformed is", rayTransformed
            intersection = self.Sensor.raycast(self.locator, origin, origin + rayTransformed*self.Sensor.rayLength)

            if intersection is not None:
                d.addLine(origin, intersection, color=[1,0,0])
            else:
                d.addLine(origin, origin+rayTransformed*self.Sensor.rayLength, color=colorMaxRange)

        vis.updatePolyData(d.getPolyData(), 'rays', colorByName='RGB255')

        #camera = self.view.camera()
        #camera.SetFocalPoint(frame.transform.GetPosition())
        #camera.SetPosition(frame.transform.TransformPoint((-30,0,10)))


    def setRobotFrameState(self, x, y, theta):
        t = vtk.vtkTransform()
        t.Translate(x,y,0.0)
        t.RotateZ(np.degrees(theta))
        self.robot.getChildFrame().copyFrame(t)

    # returns true if we are in collision
    def checkInCollision(self, raycastDistance=None):
        if raycastDistance is None:
            self.setRobotFrameState(self.Car.state[0],self.Car.state[1],self.Car.state[2])
            raycastDistance = self.Sensor.raycastAll(self.frame)

        if np.min(raycastDistance) < self.collisionThreshold:
            return True
        else:
            return False

    def tick(self):
        #print timer.elapsed
        #simulate(t.elapsed)
        x = np.sin(time.time())
        y = np.cos(time.time())
        self.setRobotFrameState(x,y,0.0)
        if (time.time() - self.playTime) > self.endTime:
            self.playTimer.stop()

    def tick2(self):
        newtime = time.time() - self.playTime
        print time.time() - self.playTime
        x = np.sin(newtime)
        y = np.cos(newtime)
        self.setRobotFrameState(x,y,0.0)

    # just increment the slider, stop the timer if we get to the end
    def playTimerCallback(self):
        self.sliderMovedByPlayTimer = True
        currentIdx = self.slider.value
        nextIdx = currentIdx + 1
        self.slider.setSliderPosition(nextIdx)
        if currentIdx >= self.sliderMax:
            print "reached end of tape, stopping playTime"
            self.playTimer.stop()

    def onSliderChanged(self, value):
        if not self.sliderMovedByPlayTimer:
            self.playTimer.stop()
        numSteps = len(self.stateOverTime)
        idx = int(np.floor(numSteps*(1.0*value/self.sliderMax)))
        idx = min(idx, numSteps-1)
        x,y,theta = self.stateOverTime[idx]
        self.setRobotFrameState(x,y,theta)
        self.sliderMovedByPlayTimer = False

    def onPlayButton(self):

        if self.playTimer.isActive():
            self.onPauseButton()
            return

        print 'play'
        self.playTimer.start()
        self.playTime = time.time()

    def onPauseButton(self):
        print 'pause'
        self.playTimer.stop()

    def plotRunData(self):

        numRuns = len(self.simulationData)
        runStart = np.zeros(numRuns)
        runDuration = np.zeros(numRuns)
        usedQValueController = np.zeros(numRuns)
        usedDefaultController = np.zeros(numRuns)
        usedDefaultRandomController= np.zeros(numRuns)
        grid = np.arange(1,numRuns+1)
        discountedReward = np.zeros(numRuns)
        avgReward = np.zeros(numRuns)


        for idx, val in enumerate(self.simulationData):
            runStart[idx] = val['startIdx']
            runDuration[idx] = val['duration']
            discountedReward[idx] = val['discountedReward']
            avgReward[idx] = val['avgReward']
            usedQValueController[idx] = (val['controllerType'] == "QValue")
            usedDefaultController[idx] = (val['controllerType'] == "default")
            usedDefaultRandomController[idx] = (val['controllerType'] == "defaultRandom")




        plt.figure(1)
        plt.subplot(3,1,1)

        idxDefaultRandom = np.where(usedDefaultRandomController==True)[0]
        idxQValueController = np.where(usedQValueController==True)[0]
        idxDefault = np.where(usedDefaultController==True)[0]

        plotData = dict()
        plotData['defaultRandom'] = {'idx': idxDefaultRandom, 'color': 'b'}
        plotData['QValue'] = {'idx': idxQValueController, 'color': 'y'}
        plotData['default'] = {'idx': idxDefault, 'color': 'g'}
        plt.title('run duration')

        for key, val in plotData.iteritems():
            plt.scatter(runStart[val['idx']], runDuration[val['idx']], color=val['color'])

        # plt.scatter(runStart[idxDefaultRandom], runDuration[idxDefaultRandom], color='b')
        # plt.scatter(runStart[idxQValueController], runDuration[idxQValueController], color='y')
        # plt.scatter(runStart[idxDefault], runDuration[idxDefault], color='g')
        plt.xlabel('run #')
        plt.ylabel('episode duration')

        plt.subplot(3,1,2)
        plt.title('discounted reward')
        for key, val in plotData.iteritems():
            plt.scatter(grid[val['idx']],discountedReward[val['idx']], color=val['color'])


        plt.subplot(3,1,3)
        plt.title("average reward")
        for key, val in plotData.iteritems():
            plt.scatter(grid[val['idx']],avgReward[val['idx']], color=val['color'])


        plt.show()




if __name__ == "__main__":
    # main(sys.argv[1:])
    parser = argparse.ArgumentParser(description='interpret simulation parameters')
    parser.add_argument('--percentObsDensity', type=float, nargs=1, default=[10])
    parser.add_argument('--endTime', type=int, nargs=1, default=[40])
    parser.add_argument('--randomizeControl', action='store_true', default=False)
    parser.add_argument('--nonRandomWorld', action='store_true', default=False)
    parser.add_argument('--circleRadius', type=float, nargs=1, default=0.7)
    parser.add_argument('--worldScale', type=float, nargs=1, default=1.0)
    parser.add_argument('--supervisedTrainingTime', type=float, nargs=1, default=500)
    argNamespace = parser.parse_args()
    percentObsDensity = argNamespace.percentObsDensity[0]
    endTime = argNamespace.endTime[0]
    randomizeControl = argNamespace.randomizeControl
    nonRandomWorld = argNamespace.nonRandomWorld
    circleRadius = argNamespace.circleRadius[0]
    worldScale = argNamespace.worldScale[0]
    supervisedTrainingTime = argNamespace.supervisedTrainingTime[0]
    sim = Simulator(percentObsDensity=percentObsDensity, endTime=endTime, randomizeControl=randomizeControl,
                    nonRandomWorld=nonRandomWorld, circleRadius=circleRadius, worldScale=worldScale,
                    supervisedTrainingTime=supervisedTrainingTime)
    sim.run()


