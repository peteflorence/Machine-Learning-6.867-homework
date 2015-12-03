__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.sarsaType = "continuous"
sim.Sarsa_numInnerBins = 4
sim.Sarsa_numOuterBins = 4
sim.Sensor_rayLength = 10

sim.randomSeed = 11
sim.randomizeControl       = True
sim.percentObsDensity      = 4
sim.nonRandomWorld         = True
sim.circleRadius           = 2.0
sim.worldScale             = 0.3
sim.supervisedTrainingTime = 5000
sim.learningTime = 4000
sim.defaultControllerTime = 0


#
#
# sim.supervisedTrainingTime = 10
# sim.learningTime = 10
# sim.defaultControllerTime = 10


sim.initialize()
sim.run()