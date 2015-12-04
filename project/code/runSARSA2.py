__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.sarsaType = "continuous"
sim.Sarsa_numInnerBins = 4
sim.Sarsa_numOuterBins = 4
sim.Sensor_rayLength = 10


# small world
sim.randomSeed = 11
sim.randomizeControl       = True
sim.percentObsDensity      = 3.5
sim.nonRandomWorld         = True
sim.circleRadius           = 2.0
sim.worldScale             = 0.5


# big world
sim.randomSeed = 8
sim.randomizeControl       = True
sim.percentObsDensity      = 4
sim.nonRandomWorld         = True
sim.circleRadius           = 2.5
sim.worldScale             = 1

sim.supervisedTrainingTime = 5000
sim.learningRandomTime = 4000
sim.learningEvalTime = 500
sim.defaultControllerTime = 500


#
#
# sim.supervisedTrainingTime = 100
# sim.learningRandomTime = 100
# sim.learningEvalTime = 100
# sim.defaultControllerTime = 100


sim.initialize()
sim.run()