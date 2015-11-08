__author__ = 'peteflorence'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

sys.path.append("../hw1")
from gradient_descent import GradientDescent


class NeuralNet:

    def __init__(self, x, t, useSoftmax=True, lam=None):
        self.N = np.size(t)

        self.M = int(self.N /10) # by default, have 1/10th number of hidden units (besides bias unit) as number of training examples

        if lam is None:
            self.lam = 1
        else:
            self.lam = lam

        self.t = np.reshape(t,(self.N,))

        # transform the k-description into a K-dimensional vector
        self.K = int(max(self.t))           # this assumes that the output labels are originally non-negative integers counting up from 1  (1, 2, ...) with no gaps
        self.T = np.zeros((self.K,self.N))
        for i in range(self.N):
            self.T[self.t[i] - 1, i] = 1

        self.x = x
        self.D = np.shape(x)[1]     # this is the dimension of the input data

        # augment the input data with ones.  this allows the bias weights to be vectorized

        ones = np.ones((self.N,1))
        self.X = np.hstack((ones,self.x)).T #note, I transposed this to make some other computation easier

        self.initializeWeights()
        self.initializeHiddenUnits()
        self.initializeOutputs()

        self.useSoftmax = useSoftmax
        if useSoftmax:
            self.sigma = self.softmax
        else:
            self.sigma = g

        if not self.useSoftmax:
            raise Exception('Currently only support gradients for softmax')

    # activation function = sigmoid
    def g(self, z):
        return 1 / (1 + np.exp(-z))

    def g_grad(self, z):
        return np.multiply(self.g(z),(1-self.g(z)))

    def softmax(self, z):
        sftmax = np.exp(z)
        sftmax = 1.0/np.sum(sftmax, axis=0)*sftmax
        return sftmax

    
    def initializeWeights(self):
        self.W1 = np.ones((self.M,self.D+1))
        self.W2 = np.ones((self.K,self.M+1))

        self.w_list = [self.W1, self.W2]

    def initializeHiddenUnits(self):
        self.a_hidden = np.zeros((self.M,1))  # activations for each unit
        self.z        = np.zeros((self.M+1,1))  # 'unit outputs'

    def initializeOutputs(self):
        self.a_outputs = np.zeros((self.K,1))  # activations for each unit
        self.y         = np.zeros((self.K,1))  # 'unit outputs'



    # not sure that this works anymore . . .
    def train(self, numiter=1):


        #need to grad descent


        #batch over all samples
        self.W1derivs = 0  # note that the derivs get summed over the for loop, but nothing else
        self.W2derivs = 0
        for i in range(self.N):
            xsample = self.X[i,:]
            tsample = self.T[i,:]

            
            self.forwardProp(xsample)
            self.calcOutputDelta(tsample)
            self.backProp()
            print np.shape(self.deltaHidden)
            self.evalDerivs(xsample)
            print np.shape(self.W1derivs)
            print np.shape(self.W2derivs)

        # grad descent update


    def forwardProp(self, xsample=None, w_list=None):


        #if W_list is None:
        #    W1 = W_list[0]

        #if W1 is None:
        #    W1 = self.W1

        if xsample is None:
            xsample = self.X


        if w_list is None:
            W1 = self.W1
            W2 = self.W2
        else:
            W1 = w_list[0]
            W2 = w_list[1]

        # FIRST LAYER
        n = np.shape(xsample)[1]
        # compute activations from weights, should be size M x n
        self.a_hidden = np.dot(W1,xsample)

        # compute output of each unit via activation function
        # should be size (M+1) x n
        self.z = np.vstack((np.ones(n),self.g(self.a_hidden)))


        # SECOND LAYER

        # compute activations from weights
        # size K x n
        self.a_outputs = np.dot(W2,self.z)

        # compute output of each unit via activation function
        # size K x 1
        self.y = self.sigma(self.a_outputs)

    # need to fill this in, will depend on the indices we are looking at
    def computeDeltaOutput(self, idx):
        self.deltaOutput = np.zeros(self.K)

    def backPropSingle(self, a_hidden):
        return np.multiply(self.g_grad(a_hidden), self.deltaOutputTimesW2)


    # what is this doing???
    def backPropFull(self):
        self.deltaOutputTimesW2 = np.dot(self.W2[:,1:].T, self.deltaOutput)
        n = np.shape(self.a_hidden)[1]
        self.deltaHidden = np.zeros((self.M,n))
        for idx in range(0,n):
            self.deltaHidden[:,idx] = np.multiply(self.g_grad(self.a_hidden[:,idx]), self.deltaOutputTimesW2[:,idx])

    def evalDerivs(self, w_list, idx=None, lam=None):
        W1 = w_list[0]
        W2 = w_list[1]

        if lam is None:
            lam = self.lam

        if idx is not None:
            n = np.size(idx)
            xsample = self.X[:,idx]
            xsample = np.reshape(xsample, (self.D+1,n))

        else:
            idx = np.arange(0,self.N)
            xsample = self.X
            n = self.N


        self.forwardProp(xsample, w_list=[W1, W2]) # should populate a_hidden, z, a_output, y
        self.computeDeltaOutput(idx) # should populate deltaOutput
        self.backPropFull()

        W1_grad = 2*lam*W1
        W2_grad = 2*lam*W2

        for i in range(0,n):
            W1_grad += np.outer(self.deltaHidden[:,i], xsample[:,i])
            W2_grad += np.outer(self.deltaOutput[:,i], self.z[:,i])


        return [W1_grad, W2_grad]



    # need to fill this in, will depend on the indices we are looking at
    # delta output should only be of size K x 1
    # turns out that this takes a very simply form, y - t, in Bishop notation
    def computeDeltaOutput(self, idx):
        if not self.useSoftmax:
            raise Exception('Currently only support gradients for softmax')

        # need to iterate over idx
        self.deltaOutput = self.y[:,idx] - self.T[:,idx]


    def evalF(self):

        # only works right now if have already forward propagated
        self.loss = 0
        sum_over_n = 0
        for i in range(self.N):
            sum_over_k = 0
            for k in range(self.K):
                sum_over_k += - self.T[i,k] * np.log(self.y) - (1 - self.T[i,k]) * np.log(1 - self.y)
            sum_over_n += sum_over_k

    def evalCost(self, lam, idx=None, w_list=None):

        # make sure we forward propagate if someone asks us to compute for specific indices
        if idx is None:
            idx = np.arange(0,self.N)
            xsample = self.X
        else:
            xsample = self.X[:,idx]

        if w_list is not None:
            W1 = w_list[0]
            W2 = w_list[1]
        else:
            W1 = self.W1
            W2 = self.W2

        # if the the user passed in an option, then we need to make sure we forward propagate in order to
        # have up-to-date information
        if (w_list is not None) or (idx is not None):
            self.forwardProp(xsample, w_list=w_list)

        loss = 0.0
        for i in range(np.size(idx)):
            loss += -np.dot(self.y[:,i], self.T[:,i])

        regTerm = lam * (np.linalg.norm(W1, ord='fro') + np.linalg.norm(W2, ord='fro'))
        loss = loss + regTerm

        return loss


    def constructGradDescentObject(self, lam=None):
        if lam is None:
            lam = self.lam


        f = lambda w_list: self.evalCost(self.lam, w_list=w_list)
        grad = lambda w_list: self.evalDerivs(w_list, lam=self.lam)
        gd = GradientDescent(f, grad=grad)
        return gd

    def plotData(self):
        if self.D != 2:
            print "can only plot data if x is two dimensional"
            return


        X_2D = self.X[1:, :]
        for i in range(1,4):
            if i==1:
                color=[1,0,0]
            if i==2:
                color=[0,1,0]
            if i==3:
                color=[0,0,1]

            idx = np.where(self.t == i)[0]
            label = " = " + str(i)
            plt.scatter(self.X[1,idx], self.X[2,idx], color=color, marker='o', facecolors='none', label=label)

        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend(loc='best')
        plt.show()

    def plotNN(self, w_list):
        self.forwardProp(self.X, w_list=w_list)

        for i in range(0,self.N):
            classIdx = np.where(self.y[:,i] == np.max(self.y[:,i]))[0]

            if classIdx==0:
                color=[1,0,0]
            if classIdx==1:
                color=[0,1,0]
            if classIdx==2:
                color=[0,0,1]

            plt.scatter(self.X[1,i], self.X[2,i], color=color, marker='o', facecolors='none')

        plt.xlabel('x_1')
        plt.ylabel('x_2')
        # plt.legend(loc='best')
        plt.show()




    

    @staticmethod
    def fromMAT(file, type="train", lam=None):
        filename = "hw3_resources/" + file + "_" + type + ".mat"
        alldata = scipy.io.loadmat(filename)['toy_data']
        X = np.array(alldata[:,0:-1])
        T = np.array(alldata[:,-1])

        nn = NeuralNet(X,T, lam=lam)
        return nn











