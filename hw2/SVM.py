__author__ = 'peteflorence'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
import cvxopt
from cvxopt.blas import dot
sys.path.append("../hw1")
from gradient_descent import GradientDescent as gd


class SVM:

    def __init__(self, x, y, kernel='linear', C=1):
        self.N = np.size(y)
        self.y = np.reshape(y,(self.N,))
        self.x = x
        self.d = np.shape(x)[1] # this is the size of w
        # pad with ones at the beginning
        self.x_full = np.zeros((self.N,self.d+1))
        self.x_full[:,0] = np.ones(self.N)
        self.x_full[:,1:] = self.x

        if kernel == 'linear':
            def k_linear(x,xprime):
                return np.inner(x,xprime)
            self.kernel = k_linear
        else: 
            self.kernel = kernel

        self.a = np.zeros((self.N, 1))[:,0] # variable for dual form optimizaiton
        self.C = C

        self.A = cvxopt.matrix(self.y, (1, self.N))

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
        return np.array(solution['x'])

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


    def predict(self, w_full, x):
        w_0 = w_full[0]
        w = w[1:]

        val = w_0 + np.dot(w,x)
        if (val > 0):
            return 1
        else:
            return 0


    def plotData(self, w_full=None):

        idx_pos = np.where(self.y > 0)
        idx_neg = np.where(self.y < 0)
        plt.scatter(self.x[idx_pos,0], self.x[idx_pos,1], color='b', marker='o', facecolors='none')
        plt.scatter(self.x[idx_neg,0], self.x[idx_neg,1], color='r', marker='o', facecolors='none')

        if w_full is not None:
            x_1_grid = np.linspace(np.min(self.x[:,0]),np.max(self.x[:,0]), 100)
            x_2_grid = -1.0/w_full[2]*(w_full[0] + w_full[1]*x_1_grid)
            plt.plot(x_1_grid, x_2_grid, color='g')

        plt.show()



    @staticmethod
    def fromFile(file, type="train"):
        filename = "hw2_resources/data/" + file + "_" + type + ".csv"
        train = np.loadtxt(filename)
        X = np.array(train[:,0:2])
        Y = np.array(train[:,2:3])
        svm = SVM(X,Y)

        return svm














