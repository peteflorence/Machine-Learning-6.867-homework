import ddapp.vtkAll as vtk
import ddapp.visualization as vis
from ddapp.debugVis import DebugData

import numpy as np

from PythonQt import QtCore, QtGui

class World(object):

    def __init__(self, worldType='simple'):
        if worldType == 'simple':
            print "Building simple world"

    @staticmethod
    def buildSimpleWorld():
        d = DebugData()
        d.addLine((2,-1,0), (2,1,0), radius=0.1)
        d.addLine((2,-1,0), (1,-2,0), radius=0.1)
        obj = vis.showPolyData(d.getPolyData(), 'world')
        return obj

    @staticmethod
    def buildBigWorld():
        d = DebugData()

        worldXmin = -20
        worldXmax = 100

        worldYmin = -50
        worldYmax = 50

        numObs = 200
        obsLength = 4

        for i in xrange(numObs):
            firstX = worldXmin + np.random.rand()*(worldXmax-worldXmin)
            firstY = worldYmin + np.random.rand()*(worldYmax-worldYmin)
            firstEndpt = (firstX,firstY,0)
            
            randTheta = np.random.rand() * 2.0*np.pi
            secondEndpt = (firstX+np.cos(randTheta), firstY+np.sin(randTheta), 0)

            d.addLine(firstEndpt, secondEndpt, radius=0.1)

        obj = vis.showPolyData(d.getPolyData(), 'world')
        return obj

    @staticmethod
    def buildRobot(x=0,y=0):

        d = DebugData()
        d.addCone((x,y,0), (1,0,0), height=0.2, radius=0.1)
        obj = vis.showPolyData(d.getPolyData(), 'robot')
        robotFrame = vis.addChildFrame(obj)
        return obj, robotFrame

    @staticmethod
    def buildCellLocator(polyData):

        loc = vtk.vtkCellLocator()
        loc.SetDataSet(polyData)
        loc.BuildLocator()
        return loc