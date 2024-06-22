from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt  
from scipy import optimize

class ProductionEconomy:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # Question 3
        par.kappa = 0.1

    def l_1(self, p1):
        ''''Optimal choice of labour for firm 1'''
        par = self.par
        l_1 = (p1*par.A*par.gamma)**(1/(1-par.gamma))
        return l_1

    def l_2(self, p2):
        ''''Optimal choice of labour for firm 2'''
        par = self.par
        l_2 = (p2*par.A*par.gamma)**(1/(1-par.gamma))
        return l_2

    def y_1(self, p1):
        '''Optimal choice of production given l and p'''
        par = self.par
        y_1 = par.A*self.l_1(p1)**par.gamma
        return y_1
    
    def y_2(self, p2):
        '''Optimal choice of production given l and p'''
        par = self.par
        y_2 = par.A*self.l_2(p2)**par.gamma
        return y_2

    def profit_1(self,p1):
        '''Profit function for firm 1'''
        par = self.par
        profit = (1-par.gamma)/par.gamma*(p1*par.A*par.gamma)**(1/(1-par.gamma))
        return profit
    
    def profit_2(self,p2):
        '''Profit function for firm 2'''
        par = self.par
        profit = (1-par.gamma)/par.gamma*(p2*par.A*par.gamma)**(1/(1-par.gamma))
        return profit
    
    def c2(self, l, p1, p2):
        '''Optimal comsumption given l'''
        par = self.par
        c2 = (1-par.alpha)*((l + self.profit_1(p1) + self.profit_2(p2))/(p2 + par.alpha*par.tau))
        return c2

    def c1(self, l, p1, p2):
        '''Optimal comsumption given l'''
        par = self.par
        c1 = par.alpha*((l + par.tau*self.c2(l, p1, p2) + self.profit_1(p1) + self.profit_2(p2))/p1)
        return c1
    
    
    def utility(self, l, p1, p2):
        '''Utility to maximize for the consumer''' 
        par = self.par
        util = par.alpha*np.log(self.c1(l, p1, p2)) + \
              (1-par.alpha)*np.log(self.c2(l, p1, p2)) - \
                  par.nu*(l**(1+par.epsilon))/(1+par.epsilon)
        return util
    
    def l_star(self, p1, p2):
        '''Returns the optimal labour supply given prices'''
        # Optimal consumer behavior
        p1 = p1
        p2 = p2
        # Define objective function to minimize as a function of l
        obj = lambda l: -self.utility(l, p1, p2)   
        bounds = [(0, None)] # bounds on labour
        l0 = 0.5 # Initial guess
        result = optimize.minimize(obj, l0, bounds = bounds, method = "SLSQP")
        l_opt = result.x[0]
        util_opt = -result.fun
        return l_opt

    def check_market_clearing(self, p1, p2):
        '''Checks the market clearing conditions'''
        par = self.par
        p1 = p1
        p2 = p2

        # Optimal firm behaviour
        l1_star = self.l_1(p1)
        l2_star = self.l_2(p2)
        y1_star = self.y_1(p1)
        y2_star = self.y_2(p2)

        l_star = self.l_star(p1, p2)
        
        # Optimal consumption for l_star
        c1_star = self.c1(l_star, p1, p2)
        c2_star = self.c2(l_star, p1, p2)
        
        # Market errors
        l_eps = l_star - l1_star - l2_star
        c1_eps = y1_star - c1_star
        c2_eps = y2_star - c2_star

        return l_eps, c1_eps, c2_eps
    
    def plot_error(self, p2_cons):
        '''Plots all market errors as a function of the prices'''
        # 1. Create p1 vector
        p1_vec = np.linspace(0.1, 2.0, 100)

        # 2. Choose a constant p2 value to vary for interactive graph
        p2_cons = p2_cons

        # 3. Lists to store errors
        e1_vals = []
        e2_vals = []
        e3_vals = []

        # 4. Looping through p1 with p2 fixed
        for p1 in p1_vec:
            e1, e2, e3 = self.check_market_clearing(p1, p2_cons)
            e1_vals.append(e1)
            e2_vals.append(e2)
            e3_vals.append(e3)

        # Convert error lists to numpy arrays for plotting
        e1_vals = np.array(e1_vals)
        e2_vals = np.array(e2_vals)
        e3_vals = np.array(e3_vals)

        # 5. Plot the errors against p1 in the same diagram
        plt.figure(figsize=(10, 6))

        # Plotting errors
        plt.plot(p1_vec, e1_vals, label='l_eps', color='b')
        plt.plot(p1_vec, e2_vals, label='c1_eps', color='g')
        plt.plot(p1_vec, e3_vals, label='c2_eps', color='r')

        # Add a horizontal line at y=0
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.xlabel('p1')
        plt.ylabel('Market error')
        plt.ylim(-1, 1)
        plt.title('Errors as a function of price')
        plt.legend()
        plt.grid(True)
        plt.show()    
    
    def market_equilibrium(self):
        '''Calculates the market equilibrium prices'''
        # 1 Objective function to minimize which determines size of the market error
        def market_error(p):
            '''Function calculates the market error by the absolute size'''
            e1, e2, e3 = self.check_market_clearing(p[0], p[1])
            sum_error = np.abs(e1) + np.abs(e2) + np.abs(e3)
            return sum_error

        # 2 Call optimzer to minimize said function
        bounds = [(0.1, 2), (0.1, 2)] # bounds on price
        p0 = [1,1] # Initial guess

        result = optimize.minimize(market_error, p0, bounds = bounds, method = "SLSQP")
        p1_opt = result.x[0]
        p2_opt = result.x[1]
        total_m_error = result.fun

        return p1_opt, p2_opt  
    
    def SWF_equilibrium(self, tax):
        par = self.par
        par.tau = tax
        
        p1_opt, p2_opt = self.market_equilibrium()
        l = self.l_star(p1_opt, p2_opt)

        c2_star = self.c2(l, p1_opt, p2_opt)
        par.T = par.tau*c2_star

        SWF_util = self.utility(l, p1_opt, p2_opt) - par.kappa * self.y_2(p2_opt)

        return SWF_util