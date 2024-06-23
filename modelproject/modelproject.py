
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

class ASADClass:

    def __init__(self, floating = True):
        """ initialize the model """

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model

        # a. externally given parameters
        par.beta_1 = 0.700 # slope of AD
        par.gamma = 0.075 # slope of SRAS
        par.beta_2 = 0.1 # Effect of interest rate on AD
        par.theta = 0.9 # Adjustment expectations for real exchange rate
        if floating == True:
            par.h = 0.5 # Central banks inflation targeting
        else:
            par.h = 0 # Fixed exchange rate economy

        # b. parameters to be chosen (guesses)
        par.delta = 0.80 # AR(1) of demand shock
        par.omega = 0.15 # AR(1) of supply shock
        par.sigma_x = 1.0 # std. of demand shock
        par.sigma_c = 0.2 # st.d of supply shock

        # c. misc paramters
        par.simT = 10_000 # length of simulation

        # d. calculate compound parameters
        self.calc_compound_par()

        # e. simulation
        sim.y_hat = np.zeros(par.simT)
        sim.pi_hat = np.zeros(par.simT)
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
        """ calculates compound parameters """

        par = self.par
        par.beta_hat = par.beta_1 + par.h*(par.beta_1/par.theta+par.beta_2)
        par.a = 1/(1+par.beta_hat*par.gamma)
        par.b = par.beta_1*(1 + par.h/par.theta)

    def simulate(self):
        """ simulate the full model """

        np.random.seed(2024)

        par = self.par
        sim = self.sim

        # a. draw random  shock innovations
        sim.x = np.random.normal(loc=0.0,scale=par.sigma_x,size=par.simT)
        sim.c = np.random.normal(loc=0.0,scale=par.sigma_c,size=par.simT)

        # b. period-by-period
        for t in range(par.simT):

            # i. lagged
            if t == 0:
                z_lag = 0.0
                s_lag = 0.0
                y_hat_lag = 0.0
                pi_hat_lag = 0.0
            else:
                z_lag = sim.z[t-1]
                s_lag = sim.s[t-1]
                y_hat_lag = sim.y_hat[t-1]
                pi_hat_lag = sim.pi_hat[t-1]

            # ii. AR(1) shocks
            z = sim.z[t] = par.delta*z_lag + sim.x[t]
            s = sim.s[t] = par.omega*s_lag + sim.c[t]

            # iii. output and inflation
            sim.y_hat[t] = (1-par.a*par.b*par.gamma)*y_hat_lag + par.a*(z-z_lag) - par.a*(par.beta_hat*s-s_lag*(par.beta_hat-par.b))
            sim.pi_hat[t] = (1-par.a*par.b*par.gamma)*pi_hat_lag + par.a*par.gamma*(s-s_lag) + par.a*par.gamma*(z-z_lag)


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
    
    def impulse_response(self, demand=False, supply=False):
        """ simulate with or without a shock """
        np.random.seed(2024)

        par = self.par
        sim = self.sim

        # a. shock demand or supply in period 0
        if demand == True:
            z_shock = 5
        else:
            z_shock = 0

        
        if supply == True:
            s_shock = 5
        else:
            s_shock = 0


        # b. period-by-period
        for t in range(par.simT):

            # shock in period 0
            if t == 0:
                z_lag = 0.0
                s_lag = 0.0
                y_hat_lag = 0.0
                pi_hat_lag = 0.0
                z = sim.z[t] = par.delta*z_lag + z_shock
                s = sim.s[t] = par.omega*s_lag + s_shock
                sim.y_hat[t] = (1-par.a*par.b*par.gamma)*y_hat_lag + par.a*(z-z_lag) - par.a*(par.beta_hat*s-s_lag*(par.beta_hat-par.b))
                sim.pi_hat[t] = (1-par.a*par.b*par.gamma)*pi_hat_lag + par.a*par.gamma*(s-s_lag) + par.a*par.gamma*(z-z_lag)

            else:
                z_lag = sim.z[t-1]
                s_lag = sim.s[t-1]
                y_hat_lag = sim.y_hat[t-1]
                pi_hat_lag = sim.pi_hat[t-1]
                z = sim.z[t] = par.delta*z_lag
                s = sim.s[t] = par.omega*s_lag
                sim.y_hat[t] = (1-par.a*par.b*par.gamma)*y_hat_lag + par.a*(z-z_lag) - par.a*(par.beta_hat*s-s_lag*(par.beta_hat-par.b))
                sim.pi_hat[t] = (1-par.a*par.b*par.gamma)*pi_hat_lag + par.a*par.gamma*(s-s_lag) + par.a*par.gamma*(z-z_lag)


def plot_ASAD(t=0, demand=True):
    '''Function to plot AS-AD model'''
    # Given parameters
    y_bar = 2
    pi_bar = 2

    beta_1 = 0.700  # slope of AD
    gamma = 0.075     # slope of SRAS
    beta_2 = 0.1    # Effect of interest rate on AD
    theta = 0.9     # Adjustment expectations for real exchange rate
    h = 0.5         # Centralbank's response to inflation gap
    beta1_hat = beta_1 + h * (beta_1 / theta + beta_2)

    y = np.linspace(-10, 10, 1000)
    
    # Initial period (t=0)
    er_0 = 0
    z_0 = 0
    s_0 = 0
    pi_AD_0 = pi_bar + beta_1 / beta1_hat * er_0 - 1 / beta_1 * (y - y_bar) + z_0 / beta_1  # AD
    pi_AS_0 = pi_bar + gamma * (y - y_bar) + s_0  # AS
    intersection_index_0 = np.argmin(np.abs(pi_AD_0 - pi_AS_0))
    
    pi_AD_list = [pi_AD_0]
    pi_AS_list = [pi_AS_0]
    er_list = [er_0]
    
    # Demand or supply shock parameters
    z = 5 if demand else 0
    s = 0 if demand else 5
    #delta, omega, sigma_x, sigma_c = res.x  # AR(1) rates that is previously calibrated

    for period in range(1, t + 1):
        z *= 0.87  # Demand shock gradually decays
        s *= 0.87  # Supply shock gradually decays

        er_prev = er_list[-1]
        pi_AS_prev = pi_AS_list[-1]
        intersection_index_prev = np.argmin(np.abs(pi_AD_list[-1] - pi_AS_prev))
        
        if demand:
            er_current = er_prev - pi_AS_prev[intersection_index_prev] + pi_bar  # ER
            pi_AD_current = pi_bar + beta_1 / beta1_hat * er_current - 1 / beta_1 * (y - y_bar) + z / beta_1  # AD
            pi_AD_list.append(pi_AD_current)
            er_list.append(er_current)
        
        pi_AS_current = pi_bar + gamma * (y - y_bar) + s  # AS
        pi_AS_list.append(pi_AS_current)

    plt.figure(figsize=(10, 6))
    plt.plot(y, pi_AD_0, linestyle="-", label="$AD_{t-0}$", color="blue")
    plt.plot(y, pi_AS_0, linestyle="-", label="$AS_{t-0}$", color="darkorange")

    for period in range(1, t + 1):
        if demand:
            plt.plot(y, pi_AD_list[period], label=f"AD$_{{t+{period}}}$", color="blue")
        plt.plot(y, pi_AS_list[period], label=f"AS$_{{t+{period}}}$", color="darkorange")
    
    plt.axvline(x=y_bar, color='red', linestyle='--', label='LRAS')
    plt.xlabel('Real GDP')
    plt.ylabel('Price Level')
    plt.legend()
    plt.title('AS-AD Diagram')