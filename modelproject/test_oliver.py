
from types import SimpleNamespace # import package SimpleNamespace. Allows for the creation of simple objects that can have arbitrary attributes set on them, making it useful for handling grouped data like parameters.

import numpy as np

class ASADClass:

    def __init__(self): #Method called when an object is created from the class, and allows the class to initialize the attributes defined under __init__
        """ initialize the model """
        #Self is a convention used to refer to the current instance of a class. When you define a method, uch as __init__ within a class, the first parameter is each method is typically named self.
        # This parameter represents the instance of the class on which the method is being called.
        # self is used to access or modify properties specific to the instance of the class, i.e. Oliver. When you create an instance of a class, python automatically passes the instance as the first argument to the method being called, and by convention, we capture that reference using self.
        par = self.par = SimpleNamespace() # parameters. This SimpleNameSpace object stores parameters, the next holds variables during simulation, the next holds statistical moments caculated from data and lastly, moms holds moments calculated from the model.
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model

        # a. externally given parameters
        par.alpha = 0.700 # slope of AD
        par.gamma = 0.075 # slope of SRAS
        par.phi = 0.99 # stickiness in expectations
        par.h = 0.3 #reaction parameter of central bank for a given deviation of the inflation target (taylor principle parameter)
        par.beta1 = 0.4 #parameter denoting the magnitude of the effect on output gap, from a deviation in real exchange rate. Set arbitrarily, can be changed
        par.beta2 = 0.1 #parameter denoting the magnitude of the effect on output gap, from a deviation in real interest rate. Set arbitrarily, can be changed

        # b. parameters to be chosen (here guesses)
        par.delta = 0.80 # AR(1) of demand shock
        par.omega = 0.15 # AR(1) of supply shock
        par.sigma_x = 1.0 # std. of demand shock
        par.sigma_c = 0.2 # st.d of supply shock

        # c. misc paramters
        par.simT = 30_000 # length of simulation

        # d. calculate compound paramters
        self.calc_compound_par()

        # e. simulation
        # Arrays to hold simulation results are initialized to zero. 
        sim.y_hat_fixed = np.zeros(par.simT)
        sim.pi_hat_fixed = np.zeros(par.simT)
        sim.y_hat_floating = np.zeros(par.simT)
        sim.pi_hat_floating = np.zeros(par.simT)
        sim.z = np.zeros(par.simT)
        sim.x = np.zeros(par.simT)
        sim.s = np.zeros(par.simT)
        sim.c = np.zeros(par.simT)

        # f. data (numbers given in notebook)
        datamoms.std_y = 1.64
        datamoms.std_pi = 0.21
        datamoms.corr_y_pi = 0.31
        datamoms.autocorr_y = 0.84
        datamoms.autocorr_pi = 0.48

    def calc_compound_par(self):
        """ calculates new compound parameters """

        par = self.par

        par.a = 1/(1+par.beta1*par.gamma)
        #par.beta1_hat = 0.8 #arbitræt sat
        par.beta1_hat = par.beta1+par.h*(par.beta1/par.phi+par.beta2)
        par.b = par.gamma*(par.beta1_hat-par.beta1)

    def simulate_fixed(self):
        """ simulate the full model """

        np.random.seed(420)

        par = self.par
        sim = self.sim

        # a. draw random  shock innovations
        sim.x = np.random.normal(loc=0.0,scale=par.sigma_x,size=par.simT) #Shock innovations are stored in the SimpleNameObject sim, under the name x
        sim.c = np.random.normal(loc=0.0,scale=par.sigma_c,size=par.simT)

        # b. period-by-period
        for t in range(par.simT): #simT=10.000

            # i. lagged
            if t == 0:
                z_lag = 0.0
                s_lag = 0.0
                y_hat_lag = 0.0
                pi_hat_lag = 0.0
            else:
                z_lag = sim.z[t-1]
                s_lag = sim.s[t-1]
                y_hat_lag = sim.y_hat_fixed[t-1]
                pi_hat_lag = sim.pi_hat_fixed[t-1]

            # ii. AR(1) shocks
            z = sim.z[t] = par.delta*z_lag + sim.x[t]
            s = sim.s[t] = par.omega*s_lag + sim.c[t]

            # iii. output and inflation
            sim.y_hat_fixed[t] = par.a*y_hat_lag + par.a*(z-z_lag) \
                - par.a*par.beta1*s 
            sim.pi_hat_fixed[t] = par.a*pi_hat_lag + par.a*(s-s_lag)+par.a*par.gamma*(z-z_lag)

    def simulate_floating(self):
        """ simulate the full model """

        np.random.seed(420)

        par = self.par
        sim = self.sim

        # a. draw random  shock innovations
        sim.x = np.random.normal(loc=0.0,scale=par.sigma_x,size=par.simT) #Shock innovations are stored in the SimpleNameObject sim, under the name x
        sim.c = np.random.normal(loc=0.0,scale=par.sigma_c,size=par.simT)

        # b. period-by-period
        for t in range(par.simT): #simT=10.000

            # i. lagged
            if t == 0:
                z_lag = 0.0
                s_lag = 0.0
                y_hat_lag = 0.0
                pi_hat_lag = 0.0
            else:
                z_lag = sim.z[t-1]
                s_lag = sim.s[t-1]
                y_hat_lag = sim.y_hat_floating[t-1]
                pi_hat_lag = sim.pi_hat_floating[t-1]

            # ii. AR(1) shocks
            z = sim.z[t] = par.delta*z_lag + sim.x[t]
            s = sim.s[t] = par.omega*s_lag + sim.c[t]

            # iii. output and inflation
            sim.y_hat_floating[t] = par.a*(1+par.b)*y_hat_lag + par.a*(z-z_lag) \
                - par.a*par.beta1_hat*s + par.alpha*s_lag*(par.beta1_hat-par.beta1)
            sim.pi_hat_floating[t] = par.a*(1+par.b)*pi_hat_lag + par.a*(s-s_lag)+par.a*par.gamma*(z-z_lag)
            
    def calc_moms(self):
        """ calculate moments """

        # note: same moments as in the data

        sim = self.sim
        moms = self.moms

        moms.std_y = np.std(sim.y_hat)
        moms.std_pi = np.std(sim.pi_hat)
        moms.corr_y_pi = np.corrcoef(sim.y_hat,sim.pi_hat)[0,1]
        moms.autocorr_y = np.corrcoef(sim.y_hat[1:],sim.y_hat[:-1])[0,1]
        moms.autocorr_pi = np.corrcoef(sim.pi_hat[1:],sim.pi_hat[:-1])[0,1]    

    def calc_diff_to_data(self,do_print=False):
        """ calculate difference to data """

        moms = self.moms
        datamoms = self.datamoms

        error = 0.0 # sum of squared differences
        for k in self.datamoms.__dict__.keys():

            diff = datamoms.__dict__[k]-moms.__dict__[k]
            error += diff**2

            if do_print: print(f'{k:12s}| data = {datamoms.__dict__[k]:.4f}, model = {moms.__dict__[k]:.4f}')

        if do_print: print(f'{error = :12.8f}')

        return error