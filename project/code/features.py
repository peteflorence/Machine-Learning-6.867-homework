__author__ = 'manuelli'


class Features(Object):

    def __init__(self):
        self.tol = 1e-3
        pass


    @staticmethod
    def computeRayFeature(rays, maxRayLength, truncationVal=):
