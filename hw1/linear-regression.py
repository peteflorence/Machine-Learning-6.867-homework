__author__ = 'manuelli'

import numpy as np
import pylab as pl


def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

class LinearRegression:

    def __init__(self,x,y,M):
        self.x= 1.0*np.array(x)
        self.y = 1.0*np.array(y)
        self.N = np.size(y)
        self.M = M
        self.initializePhi()

    def initializePhi(self):
        self.phi = np.zeros((self.N,self.M+1))
        for i in range(0,self.M+1):
            self.phi[:,i:(i+1)] = np.power(self.x,i)

    def reg(self):
        phi_pseudo_inverse = np.dot(np.linalg.inv(np.dot(np.transpose(self.phi), self.phi)),np.transpose(self.phi));
        w = np.dot(phi_pseudo_inverse, self.y)
        return w


    def SSE(self,w):
        s = np.dot(self.phi,w) - self.y
        sse = np.dot(s.transpose(),s)
        return sse

    def SSE_gradient(self,w):
        return 2.0*np.dot(self.phi.transpose(),np.dot(self.phi,w) - y)

    @staticmethod
    def getData(name):
        data = pl.loadtxt(name)
        # Returns column matrices
        X = 1.0*np.array(data[0:1].T)
        Y = 1.0*np.array(data[1:2].T)
        return X, Y

    @staticmethod
    def fromFile(filename, M):
        x,y = getData(filename)
        return LinearRegression(x,y,M)

if __name__ == "__main__":
    filename = "curvefitting.txt"
    lr = LinearRegression.fromFile(filename,3)
    w = lr.reg()
    print "w is "
    print w

    sse = lr.SSE(w)
    print "SSE is "
    print lr.SSE(w)

    grad = lr.SSE_gradient(w)
    print "SSE grad is "
    print grad



