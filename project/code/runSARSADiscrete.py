__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.sarsaType = "discrete"
sim.options['SARSA']['lam'] = 0.7
sim.options['SARSA']['numInnerBins'] = 5
sim.options['SARSA']['numOuterBins'] = 4
sim.options['SARSA']['binCutoff'] = 0.5
sim.options['Sensor']['rayLength'] = 10
sim.options['Sensor']['numRays'] = 20

sim.options['SARSA']['burnInTime'] = sim.supervisedTrainingTime/2.0
sim.options['Reward']['actionCost'] = 0.2
sim.options['Reward']['raycastCost'] = 40.0
# sim.options['Reward']['collisionPenalty'] = 200





# world from Test
sim.randomSeed = 8
sim.randomizeControl       = True
sim.percentObsDensity      = 4
sim.nonRandomWorld         = True
sim.circleRadius           = 2.5
sim.worldScale             = 1
sim.options['World']['obstaclesInnerFraction'] = 0.8


# testing
# world from Test
sim.randomSeed = 8
sim.randomizeControl       = True
sim.percentObsDensity      = 5
sim.nonRandomWorld         = True
sim.circleRadius           = 2.5
sim.worldScale             = 1


# sim.supervisedTrainingTime = 10
# sim.learningRandomTime = 10
# sim.learningEvalTime = 10
# sim.defaultControllerTime = 10

sim.initialize()
sim.run()