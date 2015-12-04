__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.sarsaType = "discrete"
sim.options['SARSA']['lam'] = 0.7
sim.options['SARSA']['useQLearningUpdate'] = True
sim.options['SARSA']['numInnerBins'] = 5
sim.options['SARSA']['numOuterBins'] = 4
sim.options['SARSA']['binCutoff'] = 0.5

sim.options['SARSA']['useSupervisedTraining'] = False


sim.options['Sensor']['rayLength'] = 10
sim.options['Sensor']['numRays'] = 20


sim.options['Reward']['actionCost'] = 0.4
sim.options['Reward']['raycastCost'] = 40.0
# sim.options['Reward']['collisionPenalty'] = 200


sim.options['Car']['velocity'] = 18

sim.options['World']['obstaclesInnerFraction'] = 0.85



# setup the training time
sim.supervisedTrainingTime = 0
sim.learningRandomTime = 4000
sim.learningEvalTime = 1000
sim.defaultControllerTime = 500


sim.options['SARSA']['burnInTime'] = sim.learningRandomTime/2.0



# World Setup
sim.randomSeed = 40
sim.randomizeControl       = True
sim.percentObsDensity      = 7.5
sim.nonRandomWorld         = True
sim.circleRadius           = 1.75
sim.worldScale             = 1




# # Testing
# sim.supervisedTrainingTime = 10
# sim.learningRandomTime = 10
# sim.learningEvalTime = 10
# sim.defaultControllerTime = 10



sim.initialize()
sim.run()