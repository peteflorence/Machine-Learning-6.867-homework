import ddapp.vtkAll as vtk
import ddapp.visualization as vis
from ddapp.debugVis import DebugData

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