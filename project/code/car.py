import numpy as np
import scipy.integrate as integrate

class CarPlant(object):

    def __init__(self, controller=None):
        # initial state
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
      
        rad = np.pi/180.0

        self.state = np.array([self.x, self.y, self.psi*rad])

        # constant velocity
        self.v = 8

        self.Controller = controller


    def dynamics(self, state, t, controlInput=None):

        dqdt = np.zeros_like(state)

        if controlInput is not None:
            u = controlInput
        else:
            # need to calculate from controller
            if self.Controller is None:
                u = np.sin(t)
            else:
                u = self.Controller.computeControlInput(state, t, self.frame)

        dqdt[0] = self.v*np.cos(state[2])
        dqdt[1] = self.v*np.sin(state[2]) 
        dqdt[2] = u
    
        return dqdt

    def setFrame(self, frame):
        self.frame = frame

    def simulate(self, dt=0.05):
        t = np.arange(0.0, 10, dt)
        newState = integrate.odeint(self.dynamics, self.state, t)
        print "Finished simulation:", newState
        print "Shape is", np.shape(newState)
        return newState

    def simulateOneStep(self, startTime=0.0, dt=0.05, controlInput=None):
        t = np.linspace(startTime, startTime+dt, 2)
        newState = integrate.odeint(self.dynamics, self.state, t, args=(controlInput,))
        self.state = newState[-1,:]
        return self.state