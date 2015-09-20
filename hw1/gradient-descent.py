__author__ = 'manuelli'
import numpy as np


class GradientDescent:

    def __init__(self, f, grad):
        self.f = f
        self.grad = grad
        if grad is None:
            self.grad = self.numericalGradient
        self.numFunctionCalls = 0
        self.numGradientCalls = 0

    def evalF(self,x):
        self.numFunctionCalls += 1
        return self.f(x)

    def evalGradient(self,x):
        self.numGradientCalls += 1
        return self.grad(x)

    def numericalGradient(self, x, dx=0.01):
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


    # x_0 is the initial guess
    def computeMin(self, x_0, stepSize = 1e-2, tol=1e-4, maxFunctionCalls=1000, useGradientCriterion=False):

        if stepSize is None:
            stepSize = 1e-1

        self.numFunctionCalls = 0
        self.numGradientCalls = 0
        x_current = x_0
        f_old = self.evalF(x_current)
        eps = 1;

        while(np.abs(eps) > tol):
            (x_current, f_current) = self.gradDescentUpdate(x_current, stepSize)
            eps = f_current - f_old
            f_old = f_current
            if useGradientCriterion:
                eps = np.max(np.abs(self.evalGradient(x_current)))


            if self.numFunctionCalls > maxFunctionCalls:
                break;

        print "x_min is "
        print x_current
        print "f_min is "
        print f_current
        print "achieved tolerance"
        print eps
        print "numFunctionCalls"
        print self.numFunctionCalls

        return (x_current, f_current, self.numFunctionCalls, tol)



    # compute one update step of gradient descent
    def gradDescentUpdate(self, x, stepSize):
        x_new = x - stepSize*self.evalGradient(x)
        f_new = self.evalF(x_new)
        return (x_new, f_new)



    @staticmethod
    def minimize(f, x_initial, grad=None, stepSize=None):
        gradDescent = GradientDescent(f, grad)
        return gradDescent.computeMin(x_initial, stepSize=stepSize)


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



