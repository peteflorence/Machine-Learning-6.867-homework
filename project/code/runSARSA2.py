__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.sarsaType = "continuous"
sim.Sarsa_numInnerBins = 4
sim.Sarsa_numOuterBins = 4
sim.Sensor_rayLength = 10


sim.randomizeControl       = True
sim.percentObsDensity      = 1.5
sim.nonRandomWorld         = True
sim.circleRadius           = 3
sim.worldScale             = 1
sim.supervisedTrainingTime = 2000
sim.learningTime = 0
sim.defaultControllerTime = 0




# sim.supervisedTrainingTime = 10
# sim.learningTime = 10
# sim.defaultControllerTime = 10


sim.initialize()
sim.run()