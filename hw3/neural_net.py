__author__ = 'peteflorence'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

sys.path.append("../hw1")
from gradient_descent import GradientDescent as gd


class NeuralNet:

    def __init__(self, x, t):
        self.N = np.size(t)

        self.M = int(self.N /10)            # by default, have 1/10th number of hidden units (besides bias unit) as number of training examples 
        self.lam = 1

        self.t = np.reshape(t,(self.N,))

        # transform the k-description into a K-dimensional vector
        self.K = int(max(self.t))           # this assumes that the output labels are originally non-negative integers counting up from 1  (1, 2, ...) with no gaps
        self.T = np.zeros((self.N,self.K))
        for i in range(self.N):
            self.T[ i , self.t[i] - 1 ] = 1

        self.x = x
        self.D = np.shape(x)[1]     # this is the dimension of the input data

        # augment the input data with ones.  this allows the bias weights to be vectorized

        ones = np.ones((self.N,1))
        self.X = np.hstack((ones,self.x))

        self.initializeWeights()
        self.initializeHiddenUnits()
        self.initializeOutputs()

    # activation function = sigmoid
    def g(self, z):
        return 1 / (1 + np.exp(-z))

    def g_grad(self, z):
        return np.multiply(self.g(z),(1-self.g(z)))

    
    def initializeWeights(self):
        self.W1 = np.ones((self.M,self.D+1))
        self.W2 = np.ones((self.K,self.M+1))

        self.w_list = [self.W1, self.W2]

    def initializeHiddenUnits(self):
        self.a_hidden = np.zeros((self.M,1))  # activations for each unit
        self.z        = np.zeros((self.M+1,1))  # 'unit outputs

    def initializeOutputs(self):
        self.a_outputs = np.zeros((self.K,1))  # activations for each unit
        self.y         = np.zeros((self.K,1))  # 'unit outputs'



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


    def forwardProp(self, xsample):
        
        # FIRST LAYER

        # compute activations from weights
        self.a_hidden = np.dot(self.W1,xsample)

        # compute output of each unit via activation function
        self.z = np.hstack((1.0,self.g(self.a_hidden)))


        # SECOND LAYER

        # compute activations from weights
        self.a_outputs = np.dot(self.W2,self.z)

        # compute output of each unit via activation function
        self.y = self.g(self.a_outputs)


    def calcOutputDelta(self, tsample):
        self.outputDelta = self.y - tsample

    def backProp(self):

        self.deltaHidden = np.multiply(self.g_grad(self.a_hidden) , np.dot(self.W2[:,1:].T,self.outputDelta))


    def evalDerivs(self, xsample):

        self.W1derivs += np.outer(xsample,self.deltaHidden).T + 2*self.lam*self.W1
        self.W2derivs += np.outer(self.z,self.outputDelta).T + 2*self.lam*self.W2

    def evalCost(self):

        # only works right now if have already forward propagated

        self.loss = 0
        sum_over_n = 0
        for i in range(self.N):
            sum_over_k = 0
            for k in range(self.K):
                sum_over_k += - self.T[i,k] * np.log(self.y) - (1 - self.T[i,k]) * np.log(1 - self.y)
            sum_over_n += sum_over_k

        self.loss = sum_over_n

        regTerm = self.lam * (np.linalg.norm(self.W1, ord='fro') + np.linalg.norm(self.W2, ord='fro'))
        self.J = self.loss + regTerm

    

    @staticmethod
    def fromMAT(file, type="train"):
        filename = "hw3_resources/" + file + "_" + type + ".mat"
        alldata = scipy.io.loadmat(filename)['toy_data']
        X = np.array(alldata[:,0:-1])
        T = np.array(alldata[:,-1])

        nn = NeuralNet(X,T)
        return nn











