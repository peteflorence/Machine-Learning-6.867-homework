__author__ = 'manuelli'
__author__ = 'manuelli'
from simulator import Simulator
import copy


options = dict()
options['SARSA'] = dict()
options['SARSA']['type'] = "discrete"
options['SARSA']['lam'] = 0.7
options['SARSA']['useQLearningUpdate'] = False
options['SARSA']['numInnerBins'] = 5
options['SARSA']['numOuterBins'] = 4
options['SARSA']['binCutoff'] = 0.5
options['SARSA']['epsilonGreedy'] = 0.4
options['SARSA']['alphaStepSize'] = 0.2

options['SARSA']['useSupervisedTraining'] = False

options['Sensor'] = dict()
options['Sensor']['rayLength'] = 10
options['Sensor']['numRays'] = 20


options['Reward'] = dict()
options['Reward']['actionCost'] = 0.4
options['Reward']['raycastCost'] = 40.0
# options['Reward']['collisionPenalty'] = 200


options['Car'] = dict()
options['Car']['velocity'] = 16

options['World'] = dict()
options['World']['obstaclesInnerFraction'] = 0.85
options['World']['randomSeed'] = 40
options['World']['percentObsDensity'] = 7.5
options['World']['nonRandomWorld'] = True
options['World']['circleRadius'] = 1.75
options['World']['scale'] = 1.0
options['dt'] = 0.05



# setup the training time
options['runTime'] = dict()
options['runTime']['supervisedTrainingTime'] = 0
options['runTime']['learningRandomTime'] = 8000
options['runTime']['learningEvalTime'] = 1500
options['runTime']['defaultControllerTime'] = 1000
#
# sim.supervisedTrainingTime = 0
# sim.learningRandomTime = 5000
# sim.learningEvalTime = 1000
# sim.defaultControllerTime = 1000


options['SARSA']['burnInTime'] = options['runTime']['learningRandomTime']/2.0



# # setup the training time
# options['runTime']['supervisedTrainingTime'] = 10
# options['runTime']['learningRandomTime'] = 20
# options['runTime']['learningEvalTime'] = 10
# options['runTime']['defaultControllerTime'] = 10




sim = Simulator(autoInitialize=False, verbose=False)
sim.options = copy.deepcopy(options)
sim.initialize()
sim.run(launchApp=True)

# Testing
#
#
#
# # sim2 = Simulator(autoInitialize=False, verbose=False)
# # sim2.options = sim.options
# # sim2.options['SARSA']['useQLearningUpdate'] = False
# #
# # sim2.initialize()
# # sim2.run()
#
#
# simList = []
# lamList = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# lamList = [0.2]
#
# for lam in lamList:
#     sim = Simulator(autoInitialize=False, verbose=False)
#     sim.options = copy.deepcopy(options)
#     sim.options['SARSA']['lam'] = lam
#     simList.append(sim)
#     sim.initialize()
#     sim.run(launchApp=False)
#     sim.plotRunData(showPlot=False)
#
#
#
# sim = simList[-1]
# sim.setupPlayback()



