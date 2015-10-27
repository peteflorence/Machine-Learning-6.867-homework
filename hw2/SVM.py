__author__ = 'peteflorence'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
import cvxopt
from cvxopt.blas import dot
sys.path.append("../hw1")
from gradient_descent import GradientDescent as gd
sys.path.append("./hw2_resources/")
from plotBoundary import plotDecisionBoundary
import math


class SVM:

    def __init__(self, x, y, kernel='linear', C=1, bandwidth=1):
        self.N = np.size(y)

        self.y = np.reshape(y,(self.N,)) * 1.0
        self.x = x * 1.0
        self.d = np.shape(x)[1] # this is the size of w

        if kernel == 'linear':
            def k_linear(x,xprime):
                return np.inner(x,xprime)
            self.kernel = k_linear
            self.kernel_type = 'linear'
        elif kernel == 'Gaussian':
            def k_gaussian(x,xprime):
                return np.exp(-np.linalg.norm(x-xprime)**2/(2*bandwidth**2))
            self.kernel = k_gaussian
            self.kernel_type = 'gaussian'
        else: 
            self.kernel = kernel
            self.kernel_type = 'not linear'

        self.a = None
        self.C = C
        self.theta = None

        self.titanicData = False

    def computeSolution(self):
        K = self.computeGramMatrix(self.x)

        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \le h
        #  Ax = b

        # define your matrices
        P = cvxopt.matrix(np.outer(self.y,self.y)*K)
        q = cvxopt.matrix(-1* np.ones(self.N))

        # inequality with 0
        # -a_i \le 0
        G_0 = cvxopt.matrix(np.diag(np.ones(self.N) * -1))
        h_0 = cvxopt.matrix(np.zeros(self.N))

        # a_i \le C
        G_slack = cvxopt.matrix(np.diag(np.ones(self.N)))
        h_slack = cvxopt.matrix(np.ones(self.N) * self.C)

        G = cvxopt.matrix(np.vstack((G_0, G_slack)))
        h = cvxopt.matrix(np.vstack((h_0, h_slack)))

        A = cvxopt.matrix(self.y, (1, self.N))
        b = cvxopt.matrix(0.0)

        # find the solution 
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        self.a = np.array(solution['x'])

        self.computeSupportVectors()
        self.computeB()
        self.computeTheta()


    def computeSupportVectors(self, threshold=1e-6):
        self.supportVectorsIdx = np.where(self.a > threshold)[0]
        self.supportVectors = self.a[self.supportVectorsIdx][:,0]
        self.NsupportVectors = self.supportVectorsIdx.size

        self.supportVectorsStrictIdx = np.where(np.logical_and(self.a < self.C - threshold, self.a > threshold))[0]
        self.supportVectorsInsideMarginIdx = np.where(self.a > self.C - threshold)[0]
        # self.supportVectorsInsideMargin = self.a[self.supportVectorsInsideMarginIdx][:,0]
        # self.NsupportVectorsInsideMargin = self.supportVectorsInsideMarginIdx

    def computeB(self, threshold=1e-6):
        # sum_over_n = 0
        # for n in range(self.NsupportVectors):
        #     sum_over_m = 0
        #     for m in range(self.NsupportVectors):
        #         sum_over_m += self.supportVectors[m]*self.y[self.supportVectorsIdx[m]]*self.kernel(self.x[self.supportVectorsIdx[n]],self.x[self.supportVectorsIdx[m]])

        #     sum_over_n += self.y[self.supportVectorsIdx[n]] - sum_over_m

        # #self.b = sum_over_n / self.NsupportVectors

        # Lucas
        # first sum over support vectors inside the margin
        sum_over_n = 0
        for idx_n in self.supportVectorsStrictIdx:
            sum_over_m = 0
            for idx_m in self.supportVectorsIdx:
                sum_over_m += self.a[idx_m]*self.y[idx_m]*self.kernel(self.x[idx_n,:], self.x[idx_m,:])

            sum_over_n += self.y[idx_n] - sum_over_m

        if np.size(self.supportVectorsStrictIdx) == 0:
            self.b = 0
        else:
            self.b = 1.0/np.size(self.supportVectorsStrictIdx)*sum_over_n




    def computeTheta(self):
        sum_over_n = 0
        for n in range(self.NsupportVectors):
            sum_over_n += self.supportVectors[n] * self.y[self.supportVectorsIdx[n]] * self.x[self.supportVectorsIdx[n]]
        self.theta = sum_over_n

    def computeGeomMargin(self):
        if self.kernel_type == 'linear':
            self.GeomMargin = 1.0 / np.linalg.norm(self.theta) 
        else:
            w_2norm_squared = 0
            for i in range(self.NsupportVectors):
                for j in range(self.NsupportVectors):
                    w_2norm_squared += self.supportVectors[i]*self.y[self.supportVectorsIdx[i]]*self.supportVectors[j]*self.y[self.supportVectorsIdx[j]]*self.kernel(self.x[self.supportVectorsIdx[i]],self.x[self.supportVectorsIdx[j]])
            w_2norm = math.sqrt(w_2norm_squared)
            self.GeomMargin = 1.0 / w_2norm


    def computeGramMatrix(self, X):
        K = np.zeros((self.N, self.N))
        
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        return K

    def softDualFormObjective(self, a, x, y, kernel):
        sum_of_a = np.sum(a)

        double_sum = 0
        for n in range(self.N):
            for m in range(self.N):
                double_sum += a[n]*a[m]*y[n]*y[m] * self.kernel(x[n],x[m])

        return sum_of_a - (1.0/2.0) * double_sum

    def aConstraint_ge0(self, a):
        # how to formulate?
        return 0

    def aConstraint_leC(self, a, C):
        # how to formulate?
        return 0

    def sumConstraint_e0(self, a, y):
        return np.matrix(a).T*np.matrix(y)


    def splitW(self,w_full):
        w_0 = w_full[0]
        w = w_full[1:]
        return (w_0,w)

    def predictorFunction(self, x):
        sum_over_n = 0
        for n in range(self.NsupportVectors):
            sum_over_n += self.supportVectors[n] * self.y[self.supportVectorsIdx[n]] * self.kernel(x,self.x[self.supportVectorsIdx[n]])

        val = sum_over_n + self.b        
        return val

    # compute the classification error rate for a given dataset
    def classificationErrorRate(self, x, y, verbose=False):

        # calculate predictor function for each x
        predict_y = y * 0.0

        for i in range(len(x)):
            predict_y[i] = self.predictorFunction(x[i,:])
        val = np.multiply(y,predict_y)

        # any entries in val that are < 0 are missclassified
        missclasified = np.size(val[val < 0])
        missclassifiedRate = missclasified/(np.size(y)*1.0)

        if verbose:
            print "number of entries missclassified = " + str(missclasified)
            print "missclassification rate  = " + str(missclassifiedRate)

        return missclassifiedRate, missclasified

    def CER_type(self,typef="train", rescale=False, verbose=False, rescaleMethod="interval", filef="stdev1"):
        if self.titanicData ==True:
            filename = "hw2_resources/data/data_titanic_" + typef + ".csv"
            T = scipy.io.loadmat(filename)['data']
            X = np.array(T[:,0:-1]) * 1.0
            Y = np.array(T[:,-1]) * 1.0
        else:
            filename = "hw2_resources/data/data_" + filef + "_" + typef + ".csv"
            train = np.loadtxt(filename)
            X = np.array(train[:,0:2]) * 1.0
            Y = np.array(train[:,2:3])[:,0] * 1.0
        if rescale:
            X = SVM.rescaleFeatures(X, method=rescaleMethod)
        missclassifiedRate, missclasified = self.classificationErrorRate(x=X, y=Y, verbose=verbose)
        return missclassifiedRate, missclasified

    def predict(self, w_full, x):
        w_0 = w_full[0]
        w = w[1:]

        val = w_0 + np.dot(w,x)
        if (val > 0):
            return 1
        else:
            return 0


    def plotData(self):
        if self.titanicData:
            print "can't plot the titanic data, it's high dimensional"
            return

        idx_pos = np.where(self.y > 0)
        idx_neg = np.where(self.y < 0)
        plt.scatter(self.x[idx_pos,0], self.x[idx_pos,1], color='b', marker='o', facecolors='none', label=' = +1')
        plt.scatter(self.x[idx_neg,0], self.x[idx_neg,1], color='r', marker='o', facecolors='none', label=' = -1')

        # intersect idx_pos and supportVectorsIdx
        idx_pos_supportVecs = np.intersect1d(idx_pos,self.supportVectorsStrictIdx)
        idx_pos_supportVecsStrict = np.intersect1d(idx_pos,self.supportVectorsInsideMarginIdx)
        plt.scatter(self.x[idx_pos_supportVecs,0], self.x[idx_pos_supportVecs,1], color='b', marker='x', s=200, facecolors='none')
        plt.scatter(self.x[idx_pos_supportVecsStrict,0], self.x[idx_pos_supportVecsStrict,1], color='b', marker='v', s=200, facecolors='none')

        idx_neg_supportVecs = np.intersect1d(idx_neg,self.supportVectorsStrictIdx)
        idx_neg_supportVecsStrict = np.intersect1d(idx_neg,self.supportVectorsInsideMarginIdx)
        plt.scatter(self.x[idx_neg_supportVecs,0], self.x[idx_neg_supportVecs,1], color='r', marker='x', s=200, facecolors='none')
        plt.scatter(self.x[idx_neg_supportVecsStrict,0], self.x[idx_neg_supportVecsStrict,1], color='r', marker='v', s=200, facecolors='none')

        if (self.theta is not None) and (self.kernel_type == 'linear'):
            w_full = np.zeros((self.d+1,1))[:,0]
            w_full[0] = self.b
            w_full[1:] = self.theta
            x_1_grid = np.linspace(np.min(self.x[:,0]),np.max(self.x[:,0]), 100)
            x_2_grid = -1.0/w_full[2]*(w_full[0] + w_full[1]*x_1_grid)
            plt.plot(x_1_grid, x_2_grid, color='g', label=' = bdry')

        elif self.a is not None:
            plotDecisionBoundary(self.x, self.y, self.predictorFunction, 0, title = "")

        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.legend(loc='best')
        plt.show()

        plt.show()



    @staticmethod
    def fromFile(file, type="train"):
        filename = "hw2_resources/data/" + file + "_" + type + ".csv"
        train = np.loadtxt(filename)
        X = np.array(train[:,0:2])
        Y = np.array(train[:,2:3])
        svm = SVM(X,Y)

        return svm

    @staticmethod
    def fromTitanic(typef="train", rescale=False, rescaleMethod="interval", kernel="linear", C=1.0, bandwidth=1.0):
        filename = "hw2_resources/data/data_titanic_" + typef + ".csv"
        T = scipy.io.loadmat(filename)['data']
        X = np.array(T[:,0:-1])
        Y = np.array(T[:,-1])
        if rescale:
            X = SVM.rescaleFeatures(X, method=rescaleMethod)
        svm = SVM(X,Y,kernel=kernel, C=C, bandwidth=bandwidth)
        svm.titanicData = True
        return svm

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

            print "yes I rescaled"


        return phiRescale











