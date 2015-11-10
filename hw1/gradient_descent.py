__author__ = 'manuelli'
import numpy as np
import time
import matplotlib.pyplot as plt


class GradientDescent:

    def __init__(self, f, grad=None):
        self.f = f
        if grad is None:
            self.grad = self.numericalGradient
        else:
            self.grad = grad


        self.stepSize = 1e-2              # learning rate
        self.tol = 1e-4

        self.numFunctionCalls = 0
        self.numGradientCalls = 0
        self.numIterations = 0

        # these will be used for stochastic gradient descent
        self.trainingData = None
        self.evalFTraining = None
        self.evalGradTraining = None

        self.initializeSGDParameters(self.stepSize)

    def initializeSGDParameters(self, stepSize, beta=2000, gamma=0.5+1e-10):
        self.beta = beta
        self.gamma = gamma
        self.alpha = stepSize*self.beta**self.gamma
        

    def evalF(self,x):
        self.numFunctionCalls += 1
        return self.f(x)

    def evalFOnTrainingData(self, idx):
        self.numFunctionCalls += 1
        return self.evalFTraining(idx)

    def evalGradient(self,x):
        self.numGradientCalls += 1
        return self.grad(x)

    def evalGradientOnTrainingData(self, x, idx):
        self.numGradientCalls += 1
        return self.evalGradTraining(x, idx)

    def numericalGradient(self, x, dx=0.00001):
        grad = np.zeros(x.shape)
        # make sure we are using floating point arithmetic, not integer
        # also need to be careful since x is pass by reference
        x_plus = 1.0*np.copy(x)
        x_minus = 1.0*np.copy(x)
        for i in xrange(0,np.size(x)):
            x_plus[i] = x_plus[i] + 1/2.0*dx
            x_minus[i] = x_minus[i] - 1/2.0*dx
            grad[i] = 1/dx*(self.evalF(x_plus) - self.evalF(x_minus))

            # reset x_plus, x_minus to original state. Saves on us having to copy all of x again
            x_plus[i] = x[i]
            x_minus[i] = x[i]

        return grad

    def computeMin(self, x_initial, maxFunctionCalls=1000, useGradientCriterion=False, storeIterValues=False, storeIterX=False, printSummary=True, tol=None):

        if x_initial is None:
            print "Please specify an initial guess"
            print "Using origin by default"
            x_initial = np.array([0.0, 0.0])

        if storeIterValues:
            self.iterValues = np.zeros((maxFunctionCalls,1))

        if tol is None:
            tol = self.tol

        self.numFunctionCalls = 0
        self.numGradientCalls = 0
        self.numIterations = 0

        x_current = x_initial
    
        f_old = self.evalF(x_current)
        eps = 1;
        startTime = time.time()

        if storeIterX:
            self.iterX = np.zeros((maxFunctionCalls,len(x_initial)))
            self.iterX[0] = x_current

        while(np.abs(eps) > tol):
            self.numIterations += 1
            (x_current, f_current) = self.gradDescentUpdate(x_current)
            eps = f_current - f_old
            f_old = f_current


            if useGradientCriterion:
                eps = np.max(np.abs(self.evalGradient(x_current)))

            if storeIterValues:
                self.iterValues[self.numIterations-1] = f_current

            if storeIterX:
                self.iterX[self.numIterations,:] = x_current

            if self.numFunctionCalls >= maxFunctionCalls:
                break;



        elapsedTime = time.time() - startTime;

        if printSummary == True:

            if self.numFunctionCalls >= maxFunctionCalls:
                print "WARNING: hit maximum number of function calls"

            print " "
            print "--- Minimization Summary --- "

            if type(x_current) != list:
                print "x_min is = " + str(x_current)

            print "f_min is = " + str(f_current)
            print "achieved tolerance = " + str(eps)
            print "numFunctionCalls = " + str(self.numFunctionCalls)
            print "optimization took " + str(elapsedTime) + " seconds"
            print "---------------------------- "
            print " "

        return (x_current, f_current, self.numFunctionCalls, self.tol)

    def plotIterValues(self):
        import matplotlib.pyplot as plt
        numIter = (self.numIterations-1)
        y_plotvalues = self.iterValues[0:numIter]
        x_plotvalues = np.linspace(1,numIter,numIter)
        plt.plot(x_plotvalues,y_plotvalues)
        plt.show()

    # compute one update step of gradient descent
    def gradDescentUpdate(self, x):
        # allow functions that take in a list
        if type(x) == list:
            # this means it is probably being called from the neural net code
            grad = self.evalGradient(x)
            x_new = []

            for idx, val in enumerate(x):
                x_new.append(x[idx] - self.stepSize*grad[idx])

            f_new = self.evalF(x_new)

        else:
            x_new = x - self.stepSize*self.evalGradient(x).T
            f_new = self.evalF(x_new)

        return (x_new, f_new)

    def stochasticGradDescentLearningRate(self, numIterations):

        return self.alpha/(numIterations + self.beta)**self.gamma


    def stochasticGradDescentUpdate(self, x, idx, numIterations):
        learningRate = self.stochasticGradDescentLearningRate(self.numIterations)
        if type(x) == list:
            x_new = []
            grad = self.evalGradientOnTrainingData(x, idx)

            for idx, val in enumerate(x):
                x_new.append(x[idx] - learningRate*grad[idx])

        else:
            x_new = x - learningRate*self.evalGradientOnTrainingData(x, idx)

        return x_new

    def stochasticGradDescent(self, x_initial, numTrainingPoints, maxFunctionCalls=10000, storeIterValues=False, printSummary=True,
                              tol=None, stepSize=0.001, useXDiffCriterion=True):
        if (self.evalGradTraining is None):
            raise Exception('you must specify evalGradTraining before running stochastic gradient descent')


        self.initializeSGDParameters(stepSize, beta=maxFunctionCalls/1.5)

        idxList = np.arange(0,numTrainingPoints)
        np.random.shuffle(idxList)

        eps = 1
        self.numFunctionCalls = 0
        self.numGradientCalls = 0
        self.numIterations = 0

        x_current = x_initial # note for neural net this will be a list of weights, each of which is a matrix
        x_old = x_initial # do this so that we get a copy
        f_old = self.evalF(x_current)
        f_current = f_old

        plotStepSizeGrid = False

        if plotStepSizeGrid:
            stepSizeGrid = np.zeros(maxFunctionCalls)

        if storeIterValues:
            self.iterValues = np.zeros(maxFunctionCalls)
            self.iterValues[0] = f_old

        if tol is None:
            tol = self.tol

        if storeIterValues or (not useXDiffCriterion):
            computeFunctionVals = True
        else:
            computeFunctionVals = False


        while(np.abs(eps) > tol):
            self.numIterations += 1
            if self.numIterations >= maxFunctionCalls:
                break
            # note I want this to be integer division, allows us to loop through the data multiple times
            idx = idxList[(self.numIterations-1) % numTrainingPoints]
            x_old = x_current
            x_current = self.stochasticGradDescentUpdate(x_current, idx, self.numIterations)

            # only compute the function values if necessary, since this is generally an expensive
            # operation that requires doing operations on the entire dataset
            if computeFunctionVals:
                f_old = f_current
                f_current = self.evalF(x_current)

            if plotStepSizeGrid:
                stepSizeGrid[self.numIterations-1] = self.stochasticGradDescentLearningRate(self.numIterations)


            if useXDiffCriterion:
                if type(x_current) == list:
                    eps=0
                    for idx, val in enumerate(x_current):
                        eps += np.linalg.norm(val-x_old[idx], ord='fro')
            else:
                eps= f_current - f_old

            if storeIterValues:
                self.iterValues[self.numIterations-1] = f_current


        f_current = self.evalF(x_current)

        if plotStepSizeGrid:
            plt.plot(np.arange(0,maxFunctionCalls), stepSizeGrid)
            plt.show()



        if printSummary == True:

            if self.numIterations >= maxFunctionCalls:
                print "WARNING: hit maximum number of function calls"

            print " "
            print "--- Minimization Summary --- "

            if type(x_current) != list:
                print "x_min is = " + str(x_current)

            print "f_min is = " + str(f_current)
            print "achieved tolerance = " + str(eps)
            print "numIterations = " + str(self.numIterations)
            print "---------------------------- "
            print " "



        # print out the results of the SGD . . .
        return x_current, f_current


    @staticmethod
    def minimize(f, x_initial, grad=None):
        gradDescent = GradientDescent(f, grad)
        return gradDescent.computeMin(x_initial)


def quad(x):
    if ~isinstance(x,np.ndarray):
        x = np.array(x)

    return np.power(x,2).sum()

def quadGrad(x):
    if ~isinstance(x,np.ndarray):
        x = np.array(x)

    return 2*x

if __name__ == "__main__":
    gd = GradientDescent(quad, quadGrad)
    x = np.array([1.0, 1.0])
    # grad = gd.numericalGradient(x)
    # print grad
    gd.computeMin(x)
    print "------------------"
    print "result using gradient convergence criterion"
    gd.computeMin(x, useGradientCriterion=True)
    # GradientDescent.minimize(quad,x,quadGrad)



