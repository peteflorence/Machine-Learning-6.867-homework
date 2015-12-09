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
import shelve

from PythonQt import QtCore, QtGui

from world import World
from car import CarPlant
from sensor import SensorObj
from controller import ControllerObj
from sarsaContinuous import SARSAContinuous
from sarsaDiscrete import SARSADiscrete
from reward import Reward
from policySearchREINFORCE import PolicySearchREINFORCE
from policySearchSGD import PolicySearchSGD


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
        self.supervisedTrainingTime = 10
        self.learningRandomTime = 10
        self.learningEvalTime = 10
        self.defaultControllerTime = 10
        self.nonRandomWorld = nonRandomWorld
        self.circleRadius = circleRadius
        self.worldScale = worldScale
        # create the visualizer object
        self.app = ConsoleApp()
        # view = app.createView(useGrid=False)
        self.view = self.app.createView(useGrid=False)

        self.initializeOptions()
        self.initializeColorMap()
        if autoInitialize:
            self.initialize()

    def initializeOptions(self):
        self.options = dict()

        self.options['Reward'] = dict()
        self.options['Reward']['actionCost'] = 0.1
        self.options['Reward']['collisionPenalty'] = 100.0
        self.options['Reward']['raycastCost'] = 20.0

        self.options['SARSA'] = dict()
        self.options['SARSA']['type'] = "discrete"
        self.options['SARSA']['lam'] = 0.7
        self.options['SARSA']['useQLearningUpdate'] = False
        self.options['SARSA']['epsilonGreedy'] = 0.2
        self.options['SARSA']['burnInTime'] = 500
        self.options['SARSA']['epsilonGreedyExponent'] = 0.3
        self.options['SARSA']['exponentialDiscountFactor'] = 0.05 #so gamma = e^(-rho*dt)
        self.options['SARSA']['numInnerBins'] = 5
        self.options['SARSA']['numOuterBins'] = 4
        self.options['SARSA']['binCutoff'] = 0.5


        self.options['World'] = dict()
        self.options['World']['obstaclesInnerFraction'] = 0.85
        self.options['World']['randomSeed'] = 40
        self.options['World']['percentObsDensity'] = 7.5
        self.options['World']['nonRandomWorld'] = True
        self.options['World']['circleRadius'] = 1.75
        self.options['World']['scale'] = 1.0


        self.options['Sensor'] = dict()
        self.options['Sensor']['rayLength'] = 10
        self.options['Sensor']['numRays'] = 20


        self.options['Car'] = dict()
        self.options['Car']['velocity'] = 12

        self.options['dt'] = 0.05


        self.options['runTime'] = dict()
        self.options['runTime']['supervisedTrainingTime'] = 10
        self.options['runTime']['learningRandomTime'] = 10
        self.options['runTime']['learningEvalTime'] = 10
        self.options['runTime']['defaultControllerTime'] = 10


    def setDefaultOptions(self):

        defaultOptions = dict()

        defaultOptions['Reward'] = dict()
        defaultOptions['Reward']['actionCost'] = 0.1
        defaultOptions['Reward']['collisionPenalty'] = 100.0
        defaultOptions['Reward']['raycastCost'] = 20.0

        defaultOptions['SARSA'] = dict()
        defaultOptions['SARSA']['type'] = "discrete"
        defaultOptions['SARSA']['lam'] = 0.7
        defaultOptions['SARSA']['useQLearningUpdate'] = False
        defaultOptions['SARSA']['useSupervisedTraining'] = True
        defaultOptions['SARSA']['epsilonGreedy'] = 0.2
        defaultOptions['SARSA']['burnInTime'] = 500
        defaultOptions['SARSA']['epsilonGreedyExponent'] = 0.3
        defaultOptions['SARSA']['exponentialDiscountFactor'] = 0.05 #so gamma = e^(-rho*dt)
        defaultOptions['SARSA']['numInnerBins'] = 5
        defaultOptions['SARSA']['numOuterBins'] = 4
        defaultOptions['SARSA']['binCutoff'] = 0.5


        defaultOptions['World'] = dict()
        defaultOptions['World']['obstaclesInnerFraction'] = 0.85
        defaultOptions['World']['randomSeed'] = 40
        defaultOptions['World']['percentObsDensity'] = 7.5
        defaultOptions['World']['nonRandomWorld'] = True
        defaultOptions['World']['circleRadius'] = 1.75
        defaultOptions['World']['scale'] = 1.0


        defaultOptions['Sensor'] = dict()
        defaultOptions['Sensor']['rayLength'] = 10
        defaultOptions['Sensor']['numRays'] = 20


        defaultOptions['Car'] = dict()
        defaultOptions['Car']['velocity'] = 12

        defaultOptions['dt'] = 0.05


        defaultOptions['runTime'] = dict()
        defaultOptions['runTime']['supervisedTrainingTime'] = 10
        defaultOptions['runTime']['learningRandomTime'] = 10
        defaultOptions['runTime']['learningEvalTime'] = 10
        defaultOptions['runTime']['defaultControllerTime'] = 10


        for k in defaultOptions:
            self.options.setdefault(k, defaultOptions[k])


        for k in defaultOptions:
            if not isinstance(defaultOptions[k], dict):
                continue

            for j in defaultOptions[k]:
                self.options[k].setdefault(j, defaultOptions[k][j])


    def initializeColorMap(self):
        self.colorMap = dict()
        self.colorMap['training'] = [1.0,  0.54901961,  0.] # this is orange
        self.colorMap['learnedEval'] = [ 0.58039216,  0.0,  0.82745098] # this is yellow
        self.colorMap['default'] = [0,1,0]

    def initialize(self):

        self.setDefaultOptions()

        self.Sensor = SensorObj(rayLength=self.options['Sensor']['rayLength'],
                                numRays=self.options['Sensor']['numRays'])
        self.Controller = ControllerObj(self.Sensor)
        self.Car = CarPlant( controller=self.Controller,
                            velocity=self.options['Car']['velocity'])
        self.Reward = Reward(self.Sensor, collisionThreshold=self.collisionThreshold,
                             actionCost=self.options['Reward']['actionCost'],
                             collisionPenalty=self.options['Reward']['collisionPenalty'],
                             raycastCost=self.options['Reward']['raycastCost'])
        
        #self.setSARSA()
        self.setPolicySearch()


        # create the things needed for simulation
        om.removeFromObjectModel(om.findObjectByName('world'))
        self.world = World.buildCircleWorld(percentObsDensity=self.options['World']['percentObsDensity'],
                                            circleRadius=self.options['World']['circleRadius'],
                                            nonRandom=self.options['World']['nonRandomWorld'],
                                            scale=self.options['World']['scale'],
                                            randomSeed=self.options['World']['randomSeed'],
                                            obstaclesInnerFraction=self.options['World']['obstaclesInnerFraction'])

        om.removeFromObjectModel(om.findObjectByName('robot'))
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

        self.supervisedTrainingTime = self.options['runTime']['supervisedTrainingTime']
        self.learningRandomTime = self.options['runTime']['learningRandomTime']
        self.learningEvalTime = self.options['runTime']['learningEvalTime']
        self.defaultControllerTime = self.options['runTime']['defaultControllerTime']

        self.Car.setFrame(self.frame)
        print "Finished initialization"

    def setSARSA(self, type=None):
        if type is None:
            type = self.options['SARSA']['type']

        if type=="discrete":
            self.Sarsa = SARSADiscrete(sensorObj=self.Sensor, actionSet=self.Controller.actionSet,
                                            collisionThreshold=self.collisionThreshold,
                                       useQLearningUpdate=self.options['SARSA']['useQLearningUpdate'],
                                       lam=self.options['SARSA']['lam'],
                                   numInnerBins = self.options['SARSA']['numInnerBins'],
                                       numOuterBins = self.options['SARSA']['numOuterBins'],
                                       binCutoff=self.options['SARSA']['binCutoff'],
                                       burnInTime = self.options['SARSA']['burnInTime'],
                                       epsilonGreedy=self.options['SARSA']['epsilonGreedy'],
                                       epsilonGreedyExponent=self.options['SARSA']['epsilonGreedyExponent'])
        elif type=="continuous":
            self.Sarsa = SARSAContinuous(sensorObj=self.Sensor, actionSet=self.Controller.actionSet,
                                         lam=self.options['SARSA']['lam'],
                                        collisionThreshold=self.collisionThreshold,
                                         burnInTime = self.options['SARSA']['burnInTime'],
                                         epsilonGreedy=self.options['SARSA']['epsilonGreedy'],
                                       epsilonGreedyExponent=self.options['SARSA']['epsilonGreedyExponent'])
        else:
            raise ValueError("sarsa type must be either discrete or continuous")

    def setPolicySearch(self):

        self.PolicySearchObj = PolicySearchSGD(sensorObj=self.Sensor, actionSet=self.Controller.actionSet,
                                        collisionThreshold=self.collisionThreshold,
                                   useQLearningUpdate=self.options['SARSA']['useQLearningUpdate'],
                                   lam=self.options['SARSA']['lam'],
                               numInnerBins = self.options['SARSA']['numInnerBins'],
                                   numOuterBins = self.options['SARSA']['numOuterBins'],
                                   binCutoff=self.options['SARSA']['binCutoff'],
                                   burnInTime = self.options['SARSA']['burnInTime'],
                                   epsilonGreedy=self.options['SARSA']['epsilonGreedy'],
                                   epsilonGreedyExponent=self.options['SARSA']['epsilonGreedyExponent'])


    def runSingleSimulation(self, updateQValues=True, controllerType='default', simulationCutoff=None, initialCarState=None):

        if initialCarState is None:
            self.setCollisionFreeInitialState()
            initialCarState = np.copy(self.Car.state)
        else:
            self.Car.state = np.copy(initialCarState)

        currentCarState = np.copy(self.Car.state)
        nextCarState = np.copy(self.Car.state)
        self.setRobotFrameState(currentCarState[0], currentCarState[1], currentCarState[2])
        currentRaycast = self.Sensor.raycastAll(self.frame)
        nextRaycast = np.zeros(self.Sensor.numRays)

        # record the reward data
        runData = dict()
        reward = 0
        discountedReward = 0
        avgReward = 0
        totalReward = 0
        startIdx = self.counter


        while (self.counter < self.numTimesteps - 1):
            idx = self.counter
            currentTime = self.t[idx]
            self.stateOverTime[idx,:] = currentCarState
            x = self.stateOverTime[idx,0]
            y = self.stateOverTime[idx,1]
            theta = self.stateOverTime[idx,2]
            self.setRobotFrameState(x,y,theta)
            currentRaycast = self.Sensor.raycastAll(self.frame)
            self.raycastData[idx,:] = currentRaycast
            S_current = (currentCarState, currentRaycast)

            if controllerType not in self.colorMap.keys():
                print
                raise ValueError("controller of type " + controllerType + " not supported")


            if controllerType in ["training","learnedEval"]:

                controlInput, controlInputIdx = self.PolicySearchObj.computeSigmoidedControlPolicy(S_current)

            if controllerType in ["default"]:
                controlInput, controlInputIdx = self.Controller.computeControlInput(currentCarState,
                                                                                currentTime, self.frame,
                                                                                raycastDistance=currentRaycast,
                                                                                randomize=False)

            self.controlInputData[idx] = controlInput

            nextCarState = self.Car.simulateOneStep(controlInput=controlInput, dt=self.dt)

            # want to compute nextRaycast so we can do the SARSA algorithm
            x = nextCarState[0]
            y = nextCarState[1]
            theta = nextCarState[2]
            self.setRobotFrameState(x,y,theta)
            nextRaycast = self.Sensor.raycastAll(self.frame)

            # compute the reward
            reward = self.Reward.computeReward(S_current, controlInput)
            self.rewardData[idx] = reward

            discountedReward += self.PolicySearchObj.gamma**(self.counter-startIdx)*reward
            totalReward += reward

            
            #bookkeeping
            currentCarState = nextCarState
            currentRaycast = nextRaycast
            self.counter+=1

            # break if we are in collision
            if self.checkInCollision(nextRaycast):
                if self.verbose: print "Had a collision, terminating simulation"
                break

            if self.counter >= simulationCutoff:
                break


        # fill in the last state by hand
        self.stateOverTime[self.counter,:] = currentCarState
        self.raycastData[self.counter,:] = currentRaycast

        # compute the average reward
        avgReward = totalReward*1.0/max(1, self.counter -startIdx)
        
        # this extra multiplication is so that it is in the same "units" as avgReward
        runData['discountedReward'] = discountedReward*(1 - self.PolicySearchObj.gamma)
        runData['avgReward'] = avgReward
        runData['totalReward'] = totalReward
        

        # this just makes sure we don't get stuck in an infinite loop.
        if startIdx == self.counter:
            self.counter += 1

        simpleReward = self.counter - startIdx

        return runData, simpleReward, initialCarState


    def runBatchSimulation(self, endTime=None, dt=0.05):


        # for use in playback
        self.dt = self.options['dt']

        self.endTime = self.learningRandomTime + self.learningEvalTime + self.defaultControllerTime

        self.t = np.arange(0.0, self.endTime, dt)
        maxNumTimesteps = np.size(self.t)
        self.stateOverTime = np.zeros((maxNumTimesteps+1, 3))
        self.raycastData = np.zeros((maxNumTimesteps+1, self.Sensor.numRays))
        self.controlInputData = np.zeros(maxNumTimesteps+1)
        self.rewardData = np.zeros(maxNumTimesteps+1)
        self.emptyQValue = np.zeros(maxNumTimesteps+1, dtype='bool')
        self.numTimesteps = maxNumTimesteps

        self.controllerTypeOrder = ['training', 'learnedEval', 'default']
        self.counter = 0
        self.simulationData = []
    
        self.initializeStatusBar()

        self.idxDict = dict()
        numRunsCounter = 0


        # three while loops for different phases of simulation: training, learnedEval, default

        loopStartIdx = self.counter
        simCutoff = min(loopStartIdx + self.supervisedTrainingTime/dt, self.numTimesteps)
        
        
        self.idxDict['training'] = self.counter
        loopStartIdx = self.counter
        simCutoff = min(loopStartIdx + self.learningRandomTime/dt, self.numTimesteps)
        
        firstTime = True

        storedReward = []
        storedTheta = []

        while ((self.counter - loopStartIdx < self.learningRandomTime/dt) and self.counter < self.numTimesteps):
            self.printStatusBar()
            startIdx = self.counter
            
            if firstTime == True:
                storedTheta.append(np.copy(self.PolicySearchObj.leftPolicy))
                runData, prevReward, initialCarState = self.runSingleSimulation(updateQValues=True, controllerType='training',
                                                   simulationCutoff=simCutoff)
                storedReward.append(np.copy(prevReward))
                
                firstTime = False

            else:
                self.PolicySearchObj.perturbOneParam()
                runData, reward, initialCarState = self.runSingleSimulation(updateQValues=True, controllerType='training',
                                                   simulationCutoff=simCutoff, initialCarState=initialCarState)
                storedReward.append(np.copy(reward))
                storedTheta.append(np.copy(self.PolicySearchObj.leftPolicy))
                
                if reward > prevReward:
                    self.PolicySearchObj.updateParams(reward, prevReward)
                    prevReward = reward
                else: 
                    self.PolicySearchObj.revertParams()
                    #initialCarState = None
                    #firstTime = True

            runData['startIdx'] = startIdx
            runData['controllerType'] = "training"
            runData['duration'] = self.counter - runData['startIdx']
            runData['endIdx'] = self.counter
            runData['runNumber'] = numRunsCounter
            numRunsCounter+=1
            self.simulationData.append(runData)

        import pickle
        savefile = open('storedReward.txt', 'w')
        pickle.dump(storedReward, savefile)
        savefile.close()

        savefile = open('storedTheta.txt', 'w')
        pickle.dump(storedTheta, savefile)
        savefile.close()



        self.idxDict['learnedEval'] = self.counter
        loopStartIdx = self.counter
        simCutoff = min(loopStartIdx + self.learningEvalTime/dt, self.numTimesteps)
        while ((self.counter - loopStartIdx < self.learningEvalTime/dt) and self.counter < self.numTimesteps):

            self.printStatusBar()
            startIdx = self.counter
            runData, reward, initialCarState = self.runSingleSimulation(updateQValues=False, controllerType='learnedEval',
                                               simulationCutoff=simCutoff)
            runData['startIdx'] = startIdx
            runData['controllerType'] = "learnedEval"
            runData['duration'] = self.counter - runData['startIdx']
            runData['endIdx'] = self.counter
            runData['runNumber'] = numRunsCounter
            numRunsCounter+=1
            self.simulationData.append(runData)



        self.idxDict['default'] = self.counter
        loopStartIdx = self.counter
        simCutoff = min(loopStartIdx + self.defaultControllerTime/dt, self.numTimesteps)
        while ((self.counter - loopStartIdx < self.defaultControllerTime/dt) and self.counter < self.numTimesteps-1):
            self.printStatusBar()
            startIdx = self.counter
            runData, reward, initialCarState = self.runSingleSimulation(updateQValues=False, controllerType='default',
                                               simulationCutoff=simCutoff)
            runData['startIdx'] = startIdx
            runData['controllerType'] = "default"
            runData['duration'] = self.counter - runData['startIdx']
            runData['endIdx'] = self.counter
            runData['runNumber'] = numRunsCounter
            numRunsCounter+=1
            self.simulationData.append(runData)


        # BOOKKEEPING
        # truncate stateOverTime, raycastData, controlInputs to be the correct size
        self.numTimesteps = self.counter + 1
        self.stateOverTime = self.stateOverTime[0:self.counter+1, :]
        self.raycastData = self.raycastData[0:self.counter+1, :]
        self.controlInputData = self.controlInputData[0:self.counter+1]
        self.rewardData = self.rewardData[0:self.counter+1]
        self.endTime = 1.0*self.counter/self.numTimesteps*self.endTime



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

    def run(self, launchApp=True):
        self.counter = 1
        self.runBatchSimulation()

        if launchApp:
            self.setupPlayback()

    def updateDrawIntersection(self, frame):

        origin = np.array(frame.transform.GetPosition())
        #print "origin is now at", origin
        d = DebugData()

        sliderIdx = self.slider.value

        controllerType = self.getControllerTypeFromCounter(sliderIdx)
        colorMaxRange = self.colorMap[controllerType]

        # if the QValue was empty then color it green
        if self.emptyQValue[sliderIdx]:
            colorMaxRange = [1,1,0] # this is yellow


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


    def getControllerTypeFromCounter(self, counter):
        name = self.controllerTypeOrder[0]

        for controllerType in self.controllerTypeOrder[1:]:
            if counter >= self.idxDict[controllerType]:
                name = controllerType


        return name


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


    def computeRunStatistics(self):
        numRuns = len(self.simulationData)
        runStart = np.zeros(numRuns)
        runDuration = np.zeros(numRuns)
        grid = np.arange(1,numRuns+1)
        discountedReward = np.zeros(numRuns)
        avgReward = np.zeros(numRuns)


        idxMap = dict()

        for controllerType, color in self.colorMap.iteritems():
            idxMap[controllerType] = np.zeros(numRuns, dtype=bool)


        for idx, val in enumerate(self.simulationData):
            runStart[idx] = val['startIdx']
            runDuration[idx] = val['duration']
            discountedReward[idx] = val['discountedReward']
            avgReward[idx] = val['avgReward']
            controllerType = val['controllerType']
            idxMap[controllerType][idx] = True


    def plotRunData(self, controllerTypeToPlot=None, showPlot=True):

        if controllerTypeToPlot==None:
            controllerTypeToPlot = self.colorMap.keys()

        numRuns = len(self.simulationData)
        runStart = np.zeros(numRuns)
        runDuration = np.zeros(numRuns)
        grid = np.arange(1,numRuns+1)
        discountedReward = np.zeros(numRuns)
        avgReward = np.zeros(numRuns)


        idxMap = dict()


        for controllerType, color in self.colorMap.iteritems():
            idxMap[controllerType] = np.zeros(numRuns, dtype=bool)


        for idx, val in enumerate(self.simulationData):
            runStart[idx] = val['startIdx']
            runDuration[idx] = val['duration']
            discountedReward[idx] = val['discountedReward']
            avgReward[idx] = val['avgReward']
            #avgRewardNoCollisionPenalty[idx] = val['avgRewardNoCollisionPenalty']
            controllerType = val['controllerType']
            idxMap[controllerType][idx] = True

            # usedQValueController[idx] = (val['controllerType'] == "QValue")
            # usedDefaultController[idx] = (val['controllerType'] == "default")
            # usedDefaultRandomController[idx] = (val['controllerType'] == "defaultRandom")
            # usedQValueRandomController[idx] = (val['controllerType'] == "QValueRandom")



        self.runStatistics = dict()
        dataMap = {'duration': runDuration, 'discountedReward':discountedReward,
                   'avgReward':avgReward}


        def computeRunStatistics(dataMap):
            for controllerType, idx in idxMap.iteritems():
                d = dict()
                for dataName, dataSet in dataMap.iteritems():
                    # average the appropriate values in dataset
                    d[dataName] = np.sum(dataSet[idx])/(1.0*np.size(dataSet[idx]))

                self.runStatistics[controllerType] = d

        computeRunStatistics(dataMap)

        if not showPlot:
            return

        plt.figure()

        #
        # idxDefaultRandom = np.where(usedDefaultRandomController==True)[0]
        # idxQValueController = np.where(usedQValueController==True)[0]
        # idxQValueControllerRandom = np.where(usedQValueControllerRandom==True)[0]
        # idxDefault = np.where(usedDefaultController==True)[0]
        #
        # plotData = dict()
        # plotData['defaultRandom'] = {'idx': idxDefaultRandom, 'color': 'b'}
        # plotData['QValue'] = {'idx': idxQValueController, 'color': 'y'}
        # plotData['default'] = {'idx': idxDefault, 'color': 'g'}

        def scatterPlot(dataToPlot):
            for controllerType in controllerTypeToPlot:
                idx = idxMap[controllerType]
                plt.scatter(grid[idx], dataToPlot[idx], color=self.colorMap[controllerType])

        def barPlot(dataName):
            plt.title(dataName)
            barWidth = 0.5
            barCounter = 0
            index = np.arange(len(controllerTypeToPlot))

            for controllerType in controllerTypeToPlot:
                val = self.runStatistics[controllerType]
                plt.bar(barCounter, val[dataName], barWidth, color=self.colorMap[controllerType], label=controllerType)
                barCounter += 1

            plt.xticks(index + barWidth/2.0, controllerTypeToPlot)


        plt.plot([1,2,3],[1,4,9])
        plt.show()

        # plt.subplot(3,1,1)
        # plt.title('run duration')
        # scatterPlot(runDuration)
        # # for controllerType, idx in idxMap.iteritems():
        # #     plt.scatter(grid[idx], runDuration[idx], color=self.colorMap[controllerType])

        # # plt.scatter(runStart[idxDefaultRandom], runDuration[idxDefaultRandom], color='b')
        # # plt.scatter(runStart[idxQValueController], runDuration[idxQValueController], color='y')
        # # plt.scatter(runStart[idxDefault], runDuration[idxDefault], color='g')
        # plt.xlabel('run #')
        # plt.ylabel('episode duration')

        # plt.subplot(3,1,2)
        # plt.title('discounted reward')
        # scatterPlot(discountedReward)
        # # for key, val in plotData.iteritems():
        # #     plt.scatter(grid[idx], discountedReward[idx], color=self.colorMap[controllerType])


        # plt.subplot(3,1,3)
        # plt.title("average reward")
        # scatterPlot(avgReward)
        # # for key, val in plotData.iteritems():
        # #     plt.scatter(grid[val['idx']],avgReward[val['idx']], color=val['color'])


        # ## plot summary statistics
        # plt.figure()

        # plt.subplot(3,1,1)
        # barPlot("duration")

        # plt.subplot(3,1,2)
        # barPlot("discountedReward")

        # plt.subplot(3,1,3)
        # barPlot("avgReward")

        # plt.show()


    def plotMultipleRunData(self, simList, toPlot=['duration', 'discountedReward'], controllerType='learned'):

        plt.figure()
        numPlots = len(toPlot)

        grid = np.arange(len(simList))
        def plot(fieldToPlot, plotNum):
            plt.subplot(numPlots,1, plotNum)
            plt.title(fieldToPlot)
            val = 0*grid
            barWidth = 0.5
            barCounter = 0
            for idx, sim in enumerate(simList):
                value = sim.runStatistics[controllerType][fieldToPlot]
                plt.bar(idx, value, barWidth)


        counter = 1
        for fieldToPlot in toPlot:
            plot(fieldToPlot, counter)
            counter += 1


        plt.show()



    def saveToFile(self, filename):
        filename = "/data/" + filename + ".out"
        my_shelf = shelve.open(filename,'n')

        my_shelf['options'] = self.options

        if self.options['SARSA']['type'] == "discrete":
            my_shelf['SARSA_QValues'] = self.Sarsa.QValues

        my_shelf.close()


    @staticmethod
    def loadFromFile():
        sim = Simulator(autoInitialize=False, verbose=False)

        my_shelf = shelve.open(filename)
        sim.options = my_shelf['options']

        sim.initialize()
        sim.Sarsa.QValues = np.array(my_shelf['SARSA_QValues'])

        my_shelf.close()

        return sim




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


