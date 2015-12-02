__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.Sarsa_numInnerBins = 4
sim.Sarsa_numOuterBins = 4

sim.randomizeControl       = True
sim.percentObsDensity      = 15
sim.nonRandomWorld         = True
sim.circleRadius           = 1.0
sim.worldScale             = 0.5
sim.supervisedTrainingTime = 3000
sim.endTime                = 9000

# sim.supervisedTrainingTime = 10
# sim.endTime                = 10

sim.initialize()
sim.Sarsa.burnIn = (sim.endTime - sim.supervisedTrainingTime) / 2.0
sim.run()