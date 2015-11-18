import numpy as np
import time
import scipy.integrate as integrate

class World(object):

    def __init__(self):
        self.buildSimpleWorld()


    def buildSimpleWorld(self):

        d = DebugData()
        d.addLine((2,-1,0), (2,1,0), radius=0.1)
        d.addLine((2,-1,0), (1,-2,0), radius=0.1)
        obj = vis.showPolyData(d.getPolyData(), 'world')
        return obj

    def buildRobot(x=0,y=0):

        d = DebugData()
        d.addCone((x,y,0), (1,0,0), height=0.2, radius=0.1)
        obj = vis.showPolyData(d.getPolyData(), 'robot')
        frame = vis.addChildFrame(obj)
        return obj


    def buildCellLocator(polyData):

        loc = vtk.vtkCellLocator()
        loc.SetDataSet(polyData)
        loc.BuildLocator()
        return loc