__author__ = 'manuelli'
from simulator import Simulator

sim = Simulator(autoInitialize=False, verbose=False)

sim.Sarsa_numInnerBins = 4
sim.Sarsa_numOuterBins = 4
sim.Sensor_rayLength = 10


sim.randomSeed = 8
sim.randomizeControl       = True
sim.percentObsDensity      = 4
sim.nonRandomWorld         = True
sim.circleRadius           = 2.5
sim.worldScale             = 1
# sim.supervisedTrainingTime = 3000
# sim.learningTime = 9000
# sim.defaultControllerTime = 1000


sim.supervisedTrainingTime = 10
sim.learningTime = 10
sim.defaultControllerTime = 10

sim.options['SARSA']['burnInTime'] = sim.supervisedTrainingTime/2.0
sim.options['Reward']['actionCost'] = 0.5
sim.initialize()
sim.run()