__author__ = 'manuelli'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

sys.path.append("../hw1")
from gradient_descent import GradientDescent as gd


class LogisticRegression:

    def __init__(self, x, y):
        self.N = np.size(y)
        self.y = np.reshape(y,(self.N,))
        self.x = x
        self.d = np.shape(x)[1] # this is the size of w
        # pad with ones at the beginning
        self.x_full = np.zeros((self.N,self.d+1))
        self.x_full[:,0] = np.ones(self.N)
        self.x_full[:,1:] = self.x
        self.titanicData = False


    def splitW(self,w_full):
        w_0 = w_full[0]
        w = w_full[1:]
        return (w_0,w)

    def NLL(self,w_full):
        (w_0, w) = self.splitW(w_full)
        exponent = -1.0*np.multiply(self.y, np.dot(self.x, w) + w_0)
        nll = np.sum(np.log(1.0 + np.exp(exponent)))

        return nll

    def NLL_Reg(self, w_full, lam):
        w = w_full[1:]
        return self.NLL(w_full) + lam*np.linalg.norm(w)**2

    # computes the gradient of NLL, not the regularization term though
    def NLL_grad(self, w_full):
        (w_0, w) = self.splitW(w_full)
        # exponent = -1.0*np.multiply(self.y, np.dot(self.x, w) + w_0)
        e_term = np.exp(-1.0*np.multiply(self.y, np.dot(self.x, w) + w_0))
        dlog = np.divide(e_term,1.0+e_term)
        a = -1.0*np.multiply(self.y[:,None], self.x_full) # this is -y x_full = N x (d+1)
        grad = np.multiply(dlog[:,None], a) # this is N x (d+1)
        grad = np.sum(grad, axis=0) # need to take column sum, now grad should be d x 1
        return grad

    # computes the gradient for the full regularized NLL objective
    def NLL_Reg_grad(self, w_full, lam):
        nllGrad = self.NLL_grad(w_full)
        regGrad = 2.0*lam*w_full # this is the gradient for the regularizer term
        regGrad[0] = 0
        return nllGrad + regGrad


    def predict(self, w_full, x):
        w_0 = w_full[0]
        w = w[1:]

        val = w_0 + np.dot(w,x)
        if (val > 0):
            return 1
        else:
            return 0

    # compute the classification error rate for a given dataset
    def classificationErrorRate(self, w_full, verbose=False):

        (w_0,w) = self.splitW(w_full)
        val = np.dot(self.x_full, w_full)
        val = np.multiply(self.y,val)

        # any entries in val that are < 0 are missclassified
        missclasified = np.size(val[val < 0])
        missclassifiedRate = missclasified/(self.N*1.0)

        if verbose:
            print "number of entries missclassified = " + str(missclasified)
            print "missclassification rate  = " + str(missclassifiedRate)

        return missclassifiedRate, missclasified

    def plotData(self, w_full=None):
        if self.titanicData:
            print "can't plot the titanic data, it's high dimensional"
            return

        idx_pos = np.where(self.y > 0)
        idx_neg = np.where(self.y < 0)
        plt.scatter(self.x[idx_pos,0], self.x[idx_pos,1], color='b', marker='o', facecolors='none', label=' = +1')
        plt.scatter(self.x[idx_neg,0], self.x[idx_neg,1], color='r', marker='o', facecolors='none', label=' = -1')

        if w_full is not None:
            x_1_grid = np.linspace(np.min(self.x[:,0]),np.max(self.x[:,0]), 100)
            x_2_grid = -1.0/w_full[2]*(w_full[0] + w_full[1]*x_1_grid)
            plt.plot(x_1_grid, x_2_grid, color='g', label=' = bdry')

        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend(loc='best')
        plt.show()

    def constructGradientDescentObject(self, lam=0):
        f = lambda w_full: self.NLL_Reg(w_full, lam)
        grad = lambda w_full: self.NLL_Reg_grad(w_full, lam)
        gradDescent = gd(f,grad=grad)
        return gradDescent

    def computeDecisionBoundary(self, w_full, lam, stepSize=0.01, maxFunctionCalls=10000, printSummary=True,
                                plot=False, plotIter=False, useGradientCriterion=False, tol=1e-4):
        gd = self.constructGradientDescentObject(lam)
        gd.stepSize = stepSize

        storeIterValues=False
        if plotIter:
            storeIterValues=True

        sol = gd.computeMin(w_full, maxFunctionCalls=maxFunctionCalls, printSummary=printSummary,
                            storeIterValues=storeIterValues, tol=tol,
                            useGradientCriterion=useGradientCriterion)
        w_star = sol[0];
        w_star_normalized = 1/np.linalg.norm(w_star)*w_star

        if printSummary:
            print "--- Classification Summary ---"
            print "w_full = " + str(w_star)
            print "w_full normalized = " + str(w_star_normalized)
            print "norm of w_full = " + str(np.linalg.norm(w_star))
            print "lambda = " + str(lam)
            self.classificationErrorRate(w_star, verbose=True)
            print "------------------"
            print ""



        if plot:
            self.plotData(w_star)

        if plotIter:
            gd.plotIterValues()

        return w_star



    @staticmethod
    def fromFile(file, type="train"):
        filename = "hw2_resources/data/" + file + "_" + type + ".csv"
        train = np.loadtxt(filename)
        X = np.array(train[:,0:2])
        Y = np.array(train[:,2:3])
        lr = LogisticRegression(X,Y)

        return lr

    @staticmethod
    def fromTitanic(type="train", rescale=False, rescaleMethod="interval"):
        filename = "hw2_resources/data/data_titanic_" + type + ".csv"
        T = scipy.io.loadmat(filename)['data']
        X = np.array(T[:,0:-1])
        Y = np.array(T[:,-1])
        if rescale:
            X = LogisticRegression.rescaleFeatures(X, method=rescaleMethod)

        lr = LogisticRegression(X,Y)
        lr.titanicData = True
        return lr

    @staticmethod
    # applies feature normalization to the matrix Phi
    def rescaleFeatures(phi, method="interval"):


        if method == "unit length":
            columnNorm = np.linalg.norm(phi, ord=2, axis=0)

            # need to be careful and remove any norms that are zero
            eps = 1e-4
            idx = np.where(columnNorm < eps)
            columnNorm[idx] = 1.0
            phiRescale = phi/columnNorm[None,:]

        if method == "interval":
            columnMin = np.min(phi, axis=0)
            columnMax = np.max(phi, axis=0)
            columnDiff = columnMax - columnMin

            eps=1e-4
            idx = np.where(columnDiff < eps)
            columnDiff[idx] = 1.0
            phiRescale = (phi - columnMin[None,:])/columnDiff[None,:]


        return phiRescale














