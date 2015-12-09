__author__ = 'manuelli'
from simulator import Simulator
import copy
import argparse



options = dict()
options['SARSA'] = dict()
options['SARSA']['type'] = "discrete"
options['SARSA']['lam'] = 0.9
options['SARSA']['useQLearningUpdate'] = True
options['SARSA']['numInnerBins'] = 5
options['SARSA']['numOuterBins'] = 4
options['SARSA']['binCutoff'] = 0.5
options['SARSA']['epsilonGreedy'] = 0.3
options['SARSA']['alphaStepSize'] = 0.2
options['SARSA']['epsilonGreedyExponent'] = 0.3
options['SARSA']['forceDriveStraight'] = True

options['SARSA']['useSupervisedTraining'] = False

options['Sensor'] = dict()
options['Sensor']['rayLength'] = 10
options['Sensor']['numRays'] = 20


options['Reward'] = dict()
options['Reward']['actionCost'] = 0.4
options['Reward']['raycastCost'] = 40.0
options['Reward']['collisionPenalty'] = 100


options['Car'] = dict()
options['Car']['velocity'] = 16

options['World'] = dict()
options['World']['obstaclesInnerFraction'] = 0.85
options['World']['randomSeed'] = 42
options['World']['percentObsDensity'] = 12
options['World']['nonRandomWorld'] = True
options['World']['circleRadius'] = 1.5
options['World']['scale'] = 1.0
options['dt'] = 0.05

# options['World'] = dict()
# options['World']['obstaclesInnerFraction'] = 0.85
# options['World']['randomSeed'] = 45
# options['World']['percentObsDensity'] = 11
# options['World']['nonRandomWorld'] = True
# options['World']['circleRadius'] = 1.5
# options['World']['scale'] = 1.0
# options['dt'] = 0.05



# setup the training time
options['runTime'] = dict()
options['runTime']['supervisedTrainingTime'] = 0
options['runTime']['learningRandomTime'] = 6500
options['runTime']['learningEvalTime'] = 1000
options['runTime']['defaultControllerTime'] = 500




options['SARSA']['burnInTime'] = options['runTime']['learningRandomTime']/(2.0*options['dt'])


filenameBase = "discrete_sarsa_test"
numRuns = 10

test=False
randomSeed = 1

if test:
    options['runTime']['supervisedTrainingTime'] = 10
    options['runTime']['learningRandomTime'] = 20
    options['runTime']['learningEvalTime'] = 10
    options['runTime']['defaultControllerTime'] = 10


simList = []
sim = Simulator(autoInitialize=False, verbose=False)
sim.options = copy.deepcopy(options)
sim.setNumpyRandomSeed(seed=randomSeed)
sim.initialize()
sim.run(launchApp=True)


