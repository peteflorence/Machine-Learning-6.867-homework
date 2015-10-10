__author__ = 'manuelli'
import numpy as np


class LogisticRegression:

    def __init__(self, x, y):
        self.N = np.size(self.y)
        self.y = np.reshape(y,(self.N,))
        self.x = x
        self.d = np.shape(x)[1] # this is the size of w


    def NLL(self,w_full):
        w_0 = w_full[0]
        w = w_full[1:]
        exponent = -1.0*numpy.multiply(self.y, np.dot(self.x, w) + w_0)
        nll = np.sum(np.log(1.0 + np.exp(exponent)))

        return nll

    def NLL_Reg(self, w_full, lam):
        w = w_full[1:]
        return self.NLL(w_full) + lam*np.norm(w)

    def predict(self, w_full, x):
        w_0 = w_full[0]
        w = w[1:]

        val = w_0 + np.dot(w,x)
        if (val > 0):
            return 1
        else:
            return 0

    def splitW(self,w_full):
        w_0 = w_full[0]
        w = w_full[1:]
        return (w_0,w)

    # compute the classification error rate for a given dataset
    def classificationErrorRate(self, w_full):

        (w_0,w) = self.splitW(w_full)
        val = w_0 + np.dot(self.x, w_full)
        val = np.multiply(y,val)

        # any entries in val that are < 0 are missclassified
        missclasified = np.size(val[val < 0])
        missclassifiedRate = missclasified/(self.N*1.0)

        return missclassifiedRate, missclasified



    @staticmethod
    def fromFile(file, type="train"):
        filename = "/hw2_resources/data/" + file + "_" + type + ".csv"
        train = loadtxt(filename)
        X = np.array(train[:,0:2])
        Y = np.array(train[:,2:3])
        lr = LogisticRegression(X,Y)

        return lr






