import numpy as np
import math

from Obstacle import Obstacle

class ObstacleField:

    def __init__(self):
        self.ObstaclesList = []
        
    def addObstacle(self,obstacle):
        self.ObstaclesList.append(obstacle)
    
    def randomField(self, M=5, mindistance = 4, maxdistance = 100):
        # M is number of obstacles 

        # -50 to 50
        # - 20 to 100

        randx = -50 + 100*np.random.rand(1,M)[0]
        randy = -20 + 120*np.random.rand(1,M)[0]

        for i in range(M):
            if math.sqrt(randx[i]**2 + randy[i]**2) > mindistance:
                obs = Obstacle(randx[i],randy[i])
                self.addObstacle(obs)

    def printObstacles(self):
        counter = 1
        for i in self.ObstaclesList:
            print "Obstacle " + str(counter) + " center:"
            i.printCenter()
            counter += 1
