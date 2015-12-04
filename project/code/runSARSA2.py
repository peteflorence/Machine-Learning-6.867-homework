__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.sarsaType = "continuous"
sim.Sarsa_numInnerBins = 4
sim.Sarsa_numOuterBins = 4
sim.Sensor_rayLength = 10

#
# # small world
# sim.randomSeed = 11
# sim.randomizeControl       = True
# sim.percentObsDensity      = 3.5
# sim.nonRandomWorld         = True
# sim.circleRadius           = 2.0
# sim.worldScale             = 0.5
#
#
# big world
# sim.randomSeed = 10
# sim.randomizeControl       = True
# sim.percentObsDensity      = 3
# sim.nonRandomWorld         = True
# sim.circleRadius           = 2.5
# sim.worldScale             = 1


# # big dense
# sim.randomSeed = 12
# sim.randomizeControl       = True
# sim.percentObsDensity      = 7
# sim.nonRandomWorld         = True
# sim.circleRadius           = 1.5
# sim.worldScale             = 1

sim.supervisedTrainingTime = 3000
sim.learningRandomTime = 4000
sim.learningEvalTime = 500
sim.defaultControllerTime = 500

sim.options['SARSA']['epsilonGreedy'] = 0.4
sim.options['SARSA']['burnInTime'] = sim.supervisedTrainingTime/1.5
sim.options['Reward']['actionCost'] = 0.4
sim.options['Reward']['raycastCost'] = 40.0

# sim.options['Reward']['collisionPenalty'] = 200


sim.randomSeed = 8
sim.randomizeControl       = True
sim.percentObsDensity      = 5
sim.nonRandomWorld         = True
sim.circleRadius           = 2.5
sim.worldScale             = 1
sim.options['World']['obstaclesInnerFraction'] = 0.8


# only try to learn the weights
sim.supervisedTrainingTime = 6000
sim.learningRandomTime = 0
sim.learningEvalTime = 1500
sim.defaultControllerTime = 0


## For testing
# sim.supervisedTrainingTime = 10
# sim.learningRandomTime = 10
# sim.learningEvalTime = 10
# sim.defaultControllerTime = 10


sim.initialize()
sim.run()