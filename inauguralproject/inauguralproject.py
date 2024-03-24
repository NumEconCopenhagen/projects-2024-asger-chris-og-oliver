from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt  
from scipy import optimize

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        '''Cobb-Douglas utility function for consumer A'''
        par = self.par
        if x1A < 0 or x2A < 0: # Ensure it does not return imaginary numbers
            util_A = 0
        else:
            util_A = x1A**(par.alpha)*x2A**(1-par.alpha)
        return util_A
        

    def utility_B(self,x1B,x2B):
        '''Cobb-Douglas utility function for consumer B'''
        par = self.par
        if x1B < 0 or x2B < 0: # Ensure it does not return imaginary numbers
            util_B = 0
        else:
            util_B = x1B**(par.beta)*x2B**(1-par.beta)

        return util_B


    def demand_A(self,p1):
        '''The optimal demand given the price'''
        par = self.par
        I_A = p1*par.w1A + par.w2A
        x1A_star = par.alpha*((I_A)/(p1))
        x2A_star = (1-par.alpha)*I_A

        return x1A_star,x2A_star


    def demand_B(self,p1):
        '''The optimal demand given the price'''
        par = self.par
        I_B = p1*par.w1B + par.w2B
        x1B_star = par.beta*((I_B)/(p1))
        x2B_star = (1-par.beta)*I_B

        return x1B_star,x2B_star
        

    def check_market_clearing(self,p1):
        '''The difference between the total optimal demand and the total market supply'''
        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A) # The difference between demand and supply for good 1
        eps2 = x2A-par.w2A + x2B-(1-par.w2A) # The difference between demand and supply for good 2

        return eps1,eps2
    

    def paretoC(self, N = 75):
        '''Pareto optimal combinations compared to initial endowment'''
        N = N # NxN amount of combinations to check
        x1A_vec = np.linspace(0,1,N)
        x2A_vec = np.linspace(0,1,N)
        uA_bar = self.utility_A(self.par.w1A, self.par.w2A) # Utility when consuming endowment for A
        uB_bar = self.utility_B(self.par.w1B, self.par.w2B) # Utility when consuming endowment for B

        kombinationer = [] # Creating empty list
        for x1a in x1A_vec:
            for x2a in x2A_vec: # looping through all combinations
                uA = self.utility_A(x1a, x2a)
                x1b = 1 - x1a
                x2b = 1 - x2a
                uB = self.utility_B(x1b, x2b)
                if uA >= uA_bar and uB >= uB_bar: # appending combinations which are Pareto efficient
                    kombinationer.append((x1a, x2a))
        return kombinationer
    
    def P_1(self, N = 75):
        '''Defines price vector for given N'''
        N = N
        P_1 = [0.5]
        i = 1
        while P_1[-1] < 2.5: # Continues while the last value is less than 2.5
            P_1.append(0.5 + (2*i)/N) # Appends each new price
            i += 1 
        return P_1


    def plot_edgeworth(self, N = 75):
        '''Plots an Edgeworth box with Pareto efficient points'''
        par = self.par

        # Call Pareto combinations from self.paretoC()
        kombinationer = self.paretoC(N)
        x1, x2 = zip(*kombinationer) # Unzipping x1 and x2 into their own vectors

        # Plot settings
        plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
        plt.rcParams.update({'font.size': 14})

        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=True,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # plotting intial endowment and pareto improvements
        ax_A.scatter(x1,x2,marker='s',color='royalblue',label='pareto improvements', s = 50)
        ax_A.scatter(self.par.w1A, self.par.w2A ,marker='s',color='black',label='endowment', s = 50)

        # limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,bbox_to_anchor=(1.1,1.0));
    
    def market_clear(self, P_1):
        '''The market errors and the market clearing price for a given price vector'''
        # Initial error values
        e1 = 10
        e2 = 10
        for p1 in P_1:
            e1_now, e2_now = self.check_market_clearing(p1)
            if np.abs(e1_now) < np.abs(e1) and np.abs(e2_now) < np.abs(e2): # Save errors and price if they are smaller in absolute value than previous
                e1 = e1_now
                e2 = e2_now
                e1_best = e1_now
                e2_best = e2_now
                p1_best = p1
        return e1_best, e2_best, p1_best

    def plot_error(self, p1, N = 75):
        '''Plots all market errors for given prices'''
        # i. market erros from P_1
        N = N
        P_1 = np.linspace(1e-4, 3, N)
        errors = []
        for p in P_1:
            e1_now, e2_now = self.check_market_clearing(p)
            errors.append((e1_now, e2_now))

        eps1, eps2 = zip(*errors)

        # Error for specific price p1
        e1, e2 = self.check_market_clearing(p1)
        
        # ii. figure
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1,1,1)
        ax.scatter(P_1,eps1,lw=2, color = "skyblue")
        ax.scatter(P_1,eps2,lw=2, color = "skyblue")
        ax.scatter(p1,e1,lw=2, label = "error1", color = "red") # Error in market 1
        ax.scatter(p1,e2,lw=2, label = "error2", color = "blue"); # Error in market 2
        ax.axhline(0, color = "black")
        ax.set_ylim([-0.5,1.5])
        ax.set_xlim([0.25,2.5])
        ax.legend()

    def A_sets_price(self, P_1):
        '''Maximizes the utility of A when dictating the price'''
        uA_0 = -np.inf # Initial utility value

        for p1 in P_1:
            # x1A and x2A defined as total market supply subtracted B's consumption
            x1_A = 1 - self.demand_B(p1)[0]
            x2_A = 1 - self.demand_B(p1)[1]

            # Loops through all utility values and saves the highest
            uA_now = self.utility_A(x1_A, x2_A)
            if uA_now > uA_0:
                uA_0 = uA_now
                uA_best = uA_now
                x1_A_best = x1_A
                x2_A_best = x2_A
                p1_A_best = p1

        return x1_A_best, x2_A_best, uA_best, p1_A_best
    
    def A_sets_price_optimize(self):
        '''Optimizer to maximize utility of A when dictating the price'''
        # a. define objective function to minimize as a function of p
        obj = lambda p: -self.utility_A(1 - self.demand_B(p)[0], 1 - self.demand_B(p)[1]) 

        # b. intitial guess and call optimizer
        p0 = 1
        result = optimize.minimize(obj, p0, method='SLSQP')

        util = -result.fun
        p = result.x[0]
        x1a = 1 - self.demand_B(p)[0]
        x2a = 1 - self.demand_B(p)[1]

        return x1a, x2a, util, p
    
    def plot_utility_A(self, P_1):
        '''Plots utility of A as well as maximum'''
        # Create an empty list to store the utility values
        utility_values = []
        A_sets_price = self.A_sets_price(P_1)

        # Iterate over each value of p1 in P_1
        for p1 in P_1:
            x1_A = 1 - self.demand_B(p1)[0]
            x2_A = 1 - self.demand_B(p1)[1]
            # Calculate the utility for the given p1
            utility = self.utility_A(x1_A, x2_A)
            utility_values.append(utility)

        # Plot the utility values
        plt.plot(P_1, utility_values, label = "utility value")
        plt.scatter(A_sets_price[3], A_sets_price[2], c="red", zorder=10, label = "maximum")  
        plt.ylabel('Utility')
        plt.xlabel('p1')
        plt.title('Utility of A for different prices')
        plt.legend()
        plt.grid(True)
        plt.show()

    def A_market_maker(self, N = 75):
        '''Maximize utility when A dictates consumption in C'''
        # Calls Pareto efficient commbinations
        kombinationer = self.paretoC(N)

        # Initial utility
        util_0 = 0

        # Iterating over possible combinatiosn and keeping for highest utility
        for i in kombinationer:
            util_now = self.utility_A(*i)
            if util_now > util_0:
                util_0 = util_now
                util_best = util_now
                x1_best, x2_best = i

        return x1_best, x2_best, util_best
    
    def A_market_maker_optimize(self):
        '''Calls optimizer utility of A with constraint that B at least gets endowment utility'''
        # a. define objective function to minimize as a function of x1 and x2
        def value_of_choice(x):
            return -self.utility_A(x[0], x[1]) 
        
        # Defines endowment utility
        uB_bar = self.utility_B(self.par.w1B, self.par.w2B)

        # b. define bounds and restrictions
        bounds = [(0, 1), (0, 1)]
        constraints = ({'type': 'eq', 'fun': lambda x: self.utility_B(1 - x[0], 1 - x[1]) - uB_bar})

        # c. intitial guess and call optimizer
        guess = [self.par.w1A, self.par.w2A]
        result = optimize.minimize(value_of_choice, guess, bounds = bounds, constraints = constraints, method='SLSQP')

        x1_best = result.x[0]
        x2_best = result.x[1]
        util_best = -result.fun

        return x1_best, x2_best, util_best
    
    def plot_edgeworth2(self, N = 75, u_a = 0.5713, u_b = 0.4865, p1 = 0.9444):
        '''Plots edgeworth box with allocations from 3)-5), indifference curves and budget line'''
        plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
        plt.rcParams.update({'font.size': 14})

        par = self.par

        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # Calls Pareto efficient points to plot them
        kombinationer = self.paretoC()
        x1, x2 = zip(*kombinationer)

        ax_A.scatter(x1,x2,marker='o',color='royalblue',label='pareto improvements')
        ax_A.scatter(par.w1A, par.w2A,marker='s',color='black',label='endowment', s = 50)
        
        # Calls price vector
        P_1 = self.P_1(N)

        # Allocations from 3-5
        # a. Market clearing allocation
        allocation_3 = self.market_clear(P_1)
        market_clearing = self.demand_A(allocation_3[2])

        # b. Price setter
        allocation_4 = self.A_sets_price(P_1)

        # c. Market maker
        allocation_5 = self.A_market_maker()

        # Plots allocations 3)-5)
        ax_A.scatter(market_clearing[0], market_clearing[1], color = "red", label = "market clearing allocation")
        ax_A.scatter(allocation_4[0], allocation_4[1], color = "yellow", label = "A is pricesetter")
        ax_A.scatter(allocation_5[0], allocation_5[1], color = "m", label = "A is market maker")

        # indifference curve for A
        u_a = u_a
        x1 = np.linspace(1e-8, 2, N)
        x2 = (u_a*x1**(-self.par.alpha))**(1/(1-self.par.alpha))

        # indifference curve for B
        u_b = u_b
        x1b = np.linspace(1e-8, 2, N)
        x2b = (u_b*x1b**(-self.par.beta))**(1/(1-self.par.beta))

        # Budget line
        p1 = p1
        x2_budget = p1*self.par.w1A + self.par.w2A - p1*x1

        ax_A.plot(x1, x2, color = "black", label = "indifference curve A")
        ax_A.plot(1-x1b, 1-x2b, color = "blue", label = "indifference curve B")
        ax_A.plot(x1, x2_budget, color = "orange", label = "budget line")

        # limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,bbox_to_anchor=(1.1,1.0));
