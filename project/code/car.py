import numpy as np
import scipy.integrate as integrate

class CarPlant(object):

    def __init__(self):
        # initial state
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
      
        rad = np.pi/180.0

        self.state = np.array([self.x, self.y, self.psi*rad])

        # constant velocity
        self.v = 8

    def dynamics(self, state, t):

        dqdt = np.zeros_like(state)
    
        # need to calculate from controller
        u = 0

        dqdt[0] = self.v*np.cos(state[2])
        dqdt[1] = self.v*np.sin(state[2]) 
        dqdt[2] = u
    
        return dqdt

    def simulate(self, dt=0.05):
    
        t = np.arange(0.0, 10, dt)
        newState = integrate.odeint(self.dynamics, self.state, t)
        print "Finished simulation:", newState
        return newState

    def simulateOneStep(self, dt=0.05):
    
        t = np.arange(0.0, dt*2, dt)
        newState = integrate.odeint(self.dynamics, self.state, t)
        return newState[-1,:]