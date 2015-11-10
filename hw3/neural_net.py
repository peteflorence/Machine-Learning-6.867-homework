__author__ = 'peteflorence'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
import time

sys.path.append("../hw1")
from gradient_descent import GradientDescent


class NeuralNet:

    def __init__(self, x, t, useSoftmax=True, lam=None, M=None):
        self.filename = None

        self.N = np.size(t)

        if M is None:
            self.M = int(self.N /10) # by default, have 1/10th number of hidden units (besides bias unit) as number of training examples
        else:
            self.M = M

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

        self.trainingTime = None

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
        self.W1 = 1.0*np.ones((self.M,self.D+1))
        self.W2 = 1.0*np.ones((self.K,self.M+1))

        self.w_list = [self.W1, self.W2]

    def initializeHiddenUnits(self):
        self.a_hidden = np.zeros((self.M,1))  # activations for each unit
        self.z        = np.zeros((self.M+1,1))  # 'unit outputs'

    def initializeOutputs(self):
        self.a_outputs = np.zeros((self.K,1))  # activations for each unit
        self.y         = np.zeros((self.K,1))  # 'unit outputs'


    def forwardProp(self, xsample=None, w_list=None):

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


    # deprecated
    def backPropSingle(self, a_hidden):
        return np.multiply(self.g_grad(a_hidden), self.deltaOutputTimesW2)


    # what is this doing???
    def backPropFull(self):
        self.deltaOutputTimesW2 = np.dot(self.W2[:,1:].T, self.deltaOutput)
        n = np.shape(self.a_hidden)[1]
        self.deltaHidden = np.zeros((self.M,n))
        
        # need to vectorize
        #for idx in range(0,n):
        #    self.deltaHidden[:,idx] = np.multiply(self.g_grad(self.a_hidden[:,idx]), self.deltaOutputTimesW2[:,idx])

        # vectorized but not for SGD
        self.deltaHidden = np.multiply(self.g_grad(self.a_hidden), self.deltaOutputTimesW2)

    def evalDerivs(self, w_list, idx=None, lam=None):
        W1 = w_list[0]
        W2 = w_list[1]

        if lam is None:
            lam = self.lam

        if idx is not None:
            idx = np.array(idx)
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


        #for i in range(0,n):
        #    W1_grad += np.outer(self.deltaHidden[:,i], xsample[:,i])
        #    W2_grad += np.outer(self.deltaOutput[:,i], self.z[:,i])

        W2_grad = np.dot(self.deltaOutput,self.z.T) # should be: KxN times Nx(M+1)
        W1_grad = np.dot(self.deltaHidden,xsample.T) # should be: MxN times Nx(D+1)


        W1_grad += 2*lam*W1
        W2_grad += 2*lam*W2

        return [W1_grad, W2_grad]


    # need to fill this in, will depend on the indices we are looking at
    # delta output should only be of size K x 1
    # turns out that this takes a very simply form, y - t, in Bishop notation
    def computeDeltaOutput(self, idx):
        if not self.useSoftmax:
            raise Exception('Currently only support gradients for softmax')

        # don't need to worry about idx, self.y and self.T are already the right size from having used
        # idx in the forwardProp step
        self.deltaOutput = self.y - self.T[:,idx]

    def evalCost(self, lam, w_list=None, skipForwardProp=False):

        xsample = self.X

        if w_list is None:
            w_list = self.w_list

        W1 = w_list[0]
        W2 = w_list[1]

        # forwardProp by default, allow user to override if they know that they have just
        # forwardProp'd with the weights they want
        if not skipForwardProp:
            self.forwardProp(xsample, w_list=w_list)


        # not working for SGD
        loss = - np.sum(np.multiply(self.T,np.log(self.y)))

        regTerm = lam * (np.linalg.norm(W1, ord='fro') + np.linalg.norm(W2, ord='fro'))
        loss = loss + regTerm

        return loss


    def constructGradDescentObject(self, lam=None):
        if lam is None:
            lam = self.lam

        f = lambda w_list: self.evalCost(lam, w_list=w_list)
        grad = lambda w_list: self.evalDerivs(w_list, lam=lam)
        gd = GradientDescent(f, grad=grad)

        def gradSGD(w_list, idx):
            idx = np.array([idx])
            lamSGD = 1.0*lam/self.N
            return self.evalDerivs(w_list, idx=idx, lam=lamSGD)

        gd.evalGradTraining = gradSGD
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


    def computeClassPrediction(self, w_list):
        self.forwardProp(self.X, w_list=w_list)
        self.classIdx = np.zeros(self.N)
        self.correctlyClassified = np.zeros(self.N)
        print self.N
        for i in range(0,self.N):
            #print i
            #print "self.y[:,i]", self.y[:,i]
            #print "np.max(self.y[:,i])", np.max(self.y[:,i])

            self.classIdx[i] = np.where(self.y[:,i] == np.max(self.y[:,i]))[0] + 1
            #print "self.classIdx[i]", self.classIdx[i]
            #print "self.t[i]", self.t[i]
            self.correctlyClassified[i] = self.classIdx[i] == self.t[i]


    def classificationErrorRate(self, w_list, verbose=False):
        self.computeClassPrediction(w_list=w_list)
        missclassifiedIdx = np.where(self.correctlyClassified==0)

        missclassified = np.size(missclassifiedIdx)
        missclassifiedRate = missclassified*1.0/self.N

        if verbose:
            print "number of entries missclassified = " + str(missclassified)
            print "missclassification rate  = " + str(missclassifiedRate)

        self.missclassifiedRate = missclassifiedRate


    def plotNN(self, w_list):
        self.computeClassPrediction(w_list=w_list)

        # for i in range(0,self.N):
        #     if self.classIdx[i]==1:
        #         color=[1,0,0]
        #         label=' =1'
        #     if self.classIdx[i]==2:
        #         color=[0,1,0]
        #         label =' =2'
        #     if self.classIdx[i]==3:
        #         color=[0,0,1]
        #         label = ' =3'
        #
        #     if not self.correctlyClassified[i]:
        #         color = [1,1,0]
        #         label = ' =misclassified'
        #
        #     plt.scatter(self.X[1,i], self.X[2,i], color=color, marker='o', facecolors='none', label=label)


        correctlyClassifiedIdx = np.where(self.correctlyClassified==1)[0]
        incorrectlyClassifiedIdx = np.where(self.correctlyClassified==0)[0]
        classIdx = []
        classIdx.append(np.intersect1d(np.where(self.classIdx == 1)[0], correctlyClassifiedIdx))
        classIdx.append(np.intersect1d(np.where(self.classIdx == 2)[0], correctlyClassifiedIdx))
        classIdx.append(np.intersect1d(np.where(self.classIdx == 3)[0], correctlyClassifiedIdx))

        X_2D = self.X[1:, :]
        for i in range(1,4):
            if i==1:
                color=[1,0,0]
            if i==2:
                color=[0,1,0]
            if i==3:
                color=[0,0,1]

            label = " = " + str(i)
            plt.scatter(self.X[1,classIdx[i-1]], self.X[2,classIdx[i-1]], color=color, marker='o', facecolors='none', label=label)

        color=[1,0.7,0]
        label = " = missclassified"
        plt.scatter(self.X[1,incorrectlyClassifiedIdx], self.X[2,incorrectlyClassifiedIdx], color=color, marker='o', facecolors='none', label=label)


        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend(loc='best')
        plt.show()


    def train(self, w_list_initial='random', useSGD=False, stepSize=0.001, maxFunctionCalls=3000, verbose=True, tol=None,
              storeIterValues=True, varname='toy_data', lam=None):
        self.reloadTrainingData(varname)
        if verbose: 
            start = time.time()
            print "Actual data"
            self.plotData()
            print 'It took', time.time()-start, 'seconds to plot original data.'


        if lam is None:
            lam = self.lam

        
        start = time.time()
        gd = self.constructGradDescentObject(lam=lam)

        gd.stepSize = stepSize
        
        if w_list_initial =='random':
            scale=0.5
            w_initial = [scale*(np.random.random_sample(np.shape(self.W1)) - 0.5 ), scale*(np.random.random_sample(np.shape(self.W2)) - 0.5)]


        if useSGD:
            print "using STOCHASTIC gradient descent"
            if storeIterValues:
                print ""
                print "---------------------"
                print "WARNING: You are storing function values while using SGD"
                print "this will significantly slow down the optimization"
                print "---------------------"
                print " "
            w_min, f_min, = gd.stochasticGradDescent(w_initial, self.N, maxFunctionCalls=maxFunctionCalls,
                                                     storeIterValues=storeIterValues, printSummary=verbose, tol=tol)
        else:
            print "using standard gradient descent"
            w_min, f_min, _, _ = gd.computeMin(w_initial, maxFunctionCalls=maxFunctionCalls, storeIterValues=True, printSummary=verbose)

        if verbose and storeIterValues:
            gd.plotIterValues()

        self.trainingTime = time.time()-start
        print 'It took', self.trainingTime, 'seconds to train.'

        if verbose: 
            start = time.time()
            print "Neural net classifier"
            self.plotNN(w_min)
            print 'It took', time.time()-start, 'seconds to plot classification predictions.'

        self.classificationErrorRate(w_min, verbose=True)

        # Save optimal weights to the object variables
        self.W1 = w_min[0]
        self.W2 = w_min[1]

    def loadAnotherDataset(self, filename, varname):
        alldata = scipy.io.loadmat(filename)[varname]
        x = np.array(alldata[:,0:-1])
        t = np.array(alldata[:,-1])
        self.N = np.size(t)
        self.t = np.reshape(t,(self.N,))

        # transform the k-description into a K-dimensional vector
        self.T = np.zeros((self.K,self.N))
        for i in range(self.N):
            self.T[self.t[i] - 1, i] = 1

        self.x = x
        self.D = np.shape(x)[1]     # this is the dimension of the input data

        # augment the input data with ones.  this allows the bias weights to be vectorized
        ones = np.ones((self.N,1))
        self.X = np.hstack((ones,self.x)).T #note, I transposed this to make some other computation easier


    def test(self, verbose=True, varname='toy_data'):
        print "### TEST DATASET ###"
        start = time.time()
        
        filename = "hw3_resources/" + self.filename + "_" + "test" + ".mat"
        self.loadAnotherDataset(filename, varname)

        if verbose:
            self.plotNN([self.W1, self.W2])
        self.classificationErrorRate([self.W1, self.W2], verbose=True)

        print 'It took', time.time()-start, 'seconds to test.'

    def validate(self, verbose=True, varname='toy_data'):
        print "### VALIDATION DATASET ###"
        start = time.time()
        
        filename = "hw3_resources/" + self.filename + "_" + "validate" + ".mat"
        self.loadAnotherDataset(filename, varname)

        if verbose:
            self.plotNN([self.W1, self.W2])
        self.classificationErrorRate([self.W1, self.W2], verbose=True)

        print 'It took', time.time()-start, 'seconds to validate.'

    def reloadTrainingData(self, varname='toy_data'):

        filename = "hw3_resources/" + self.filename + "_" + "train" + ".mat"
        self.loadAnotherDataset(filename, varname)


    @staticmethod
    def fromMAT(file, varname='toy_data', type="train", lam=None, M=None):
        filename = "hw3_resources/" + file + "_" + type + ".mat"
        alldata = scipy.io.loadmat(filename)[varname]
        X = np.array(alldata[:,0:-1])
        T = np.array(alldata[:,-1])


        nn = NeuralNet(X,T, lam=lam, M=M)
        nn.filename = file
        return nn





