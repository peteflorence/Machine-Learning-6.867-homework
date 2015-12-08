__author__ = 'manuelli'
from simulator import Simulator
import copy
import argparse



options = dict()
options['SARSA'] = dict()
options['SARSA']['type'] = "discrete"
options['SARSA']['lam'] = 0.7
options['SARSA']['useQLearningUpdate'] = True
options['SARSA']['numInnerBins'] = 5
options['SARSA']['numOuterBins'] = 4
options['SARSA']['binCutoff'] = 0.5
options['SARSA']['epsilonGreedy'] = 0.4
options['SARSA']['alphaStepSize'] = 0.2
options['SARSA']['epsilonGreedyExponent'] = 0.3

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
options['runTime']['learningRandomTime'] = 6500
options['runTime']['learningEvalTime'] = 2000
options['runTime']['defaultControllerTime'] = 1000




# # setup the training time
# options['runTime']['supervisedTrainingTime'] = 10
# options['runTime']['learningRandomTime'] = 20
# options['runTime']['learningEvalTime'] = 10
# options['runTime']['defaultControllerTime'] = 10


options['SARSA']['burnInTime'] = options['runTime']['learningRandomTime']/(2.0*options['dt'])


filenameBase = "discrete_QLearning_lam_"
lamList = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
# lamList = [0.0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='interpret simulation parameters')
    parser.add_argument('--runSim', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    argNamespace = parser.parse_args()
    runSim = argNamespace.runSim
    test = argNamespace.test

if test:
    options['runTime']['supervisedTrainingTime'] = 10
    options['runTime']['learningRandomTime'] = 20
    options['runTime']['learningEvalTime'] = 10
    options['runTime']['defaultControllerTime'] = 10

if runSim:
    print "WARNING!!!!"
    print "running simulation, break now to avoid overwriting files!!!!"


simList = []

for lam in lamList:
    filename = filenameBase + str(lam)

    if runSim:
        sim = Simulator(autoInitialize=False, verbose=False)
        sim.options = copy.deepcopy(options)
        sim.options['SARSA']['lam'] = lam
        sim.initialize()
        sim.run(launchApp=False)
        sim.saveToFile(filename)

    else:
        sim = Simulator.loadFromFile(filename)
        simList.append(sim)

# om.removeFromObjectModel(om.findObjectByName("world"))
# om.removeFromObjectModel(om.findObjectByName("robot"))
# simList[0].initialize()

if not runSim:
    sim = simList[0]
    sim.setupPlayback()


