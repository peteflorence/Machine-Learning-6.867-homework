__author__ = 'manuelli'
import numpy as np


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
        

    def evalF(self,x):
        self.numFunctionCalls += 1
        return self.f(x)

    def evalGradient(self,x):
        self.numGradientCalls += 1
        return self.grad(x)

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

    def computeMin(self, x_initial, maxFunctionCalls=1000, useGradientCriterion=False, storeIterValues=False, storeIterX=False, printSummary=True):

        if x_initial is None:
            print "Please specify an initial guess"
            print "Using origin by default"
            x_initial = np.array([0.0, 0.0])

        if storeIterValues:
            self.iterValues = np.zeros((maxFunctionCalls,1))




        self.numFunctionCalls = 0
        self.numGradientCalls = 0
        self.numIterations = 0

        x_current = x_initial
    
        f_old = self.evalF(x_current)
        eps = 1;

        if storeIterX:
            self.iterX = np.zeros((maxFunctionCalls,len(x_initial)))
            self.iterX[0] = x_current

        while(np.abs(eps) > self.tol):
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




        if printSummary == True:

            if self.numFunctionCalls >= maxFunctionCalls:
                print "WARNING: hit maximum number of function calls"

            print " "
            print "--- Minimization Summary --- "
            print "x_min is = " + str(x_current)
            print "f_min is = " + str(f_current)
            print "achieved tolerance = " + str(eps)
            print "numFunctionCalls = " + str(self.numFunctionCalls)
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
        x_new = x - self.stepSize*self.evalGradient(x).T
        f_new = self.evalF(x_new)
        return (x_new, f_new)

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



