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
        par = self.par
        if x1A < 0 or x2A < 0:
            util_A = 0
        else:
            util_A = x1A**(par.alpha)*x2A**(1-par.alpha)
        return util_A
        

    def utility_B(self,x1B,x2B):
        par = self.par
        if x1B < 0 or x2B < 0:
            util_B = 0
        else:
            util_B = x1B**(par.beta)*x2B**(1-par.beta)

        return util_B

    def demand_A(self,p1):
        par = self.par
        I_A = p1*par.w1A + par.w2A
        x1A_star = par.alpha*((I_A)/(p1))
        x2A_star = (1-par.alpha)*I_A

        return x1A_star,x2A_star

    def demand_B(self,p1):
        par = self.par
        I_B = p1*par.w1B + par.w2B
        x1B_star = par.beta*((I_B)/(p1))
        x2B_star = (1-par.beta)*I_B

        return x1B_star,x2B_star
        

    def check_market_clearing(self,p1):
        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def paretoC(self, x1A_vec, x2A_vec, uA_bar, uB_bar):
        kombinationer = []
        for x1a in x1A_vec:
            for x2a in x2A_vec:
                uA = self.utility_A(x1a, x2a)
                x1b = 1 - x1a
                x2b = 1 - x2a
                uB = self.utility_B(x1b, x2b)
                if uA >= uA_bar and uB >= uB_bar:
                    kombinationer.append((x1a, x2a))
        return kombinationer
    
    def market_clear(self, P_1):
        e1 = 10
        e2 = 10
        for p1 in P_1:
            e1_now, e2_now = self.check_market_clearing(p1)
            if np.abs(e1_now) < np.abs(e1) and np.abs(e2_now) < np.abs(e2):
                e1 = e1_now
                e2 = e2_now
                e1_best = e1_now
                e2_best = e2_now
                p1_best = p1
        return e1_best, e2_best, p1_best

    def plot_error(self, p1, N):
        # i. market erros from P_1
        N = N
        P_1 = np.linspace(1e-4, 3, N)
        errors = []
        for p in P_1:
            e1_now, e2_now = self.check_market_clearing(p)
            errors.append((e1_now, e2_now))

        eps1, eps2 = zip(*errors)

        e1, e2 = self.check_market_clearing(p1)
        
        # ii. figure
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1,1,1)
        ax.scatter(P_1,eps1,lw=2, color = "skyblue")
        ax.scatter(P_1,eps2,lw=2, color = "skyblue")
        ax.scatter(p1,e1,lw=2, label = "error1", color = "red")
        ax.scatter(p1,e2,lw=2, label = "error2", color = "blue");
        ax.axhline(0, color = "black")
        ax.set_ylim([-0.5,1.5])
        ax.set_xlim([0.25,2.5])
        ax.legend()

    def A_sets_price(self, P_1):
        uA_0 = -np.inf
        for p1 in P_1:
            x1_A = 1 - self.demand_B(p1)[0]
            x2_A = 1 - self.demand_B(p1)[1]
            uA_now = self.utility_A(x1_A, x2_A)
            if uA_now > uA_0:
                uA_0 = uA_now
                uA_best = uA_now
                x1_A_best = x1_A
                x2_A_best = x2_A
                p1_A_best = p1

        return x1_A_best, x2_A_best, uA_best, p1_A_best
    
    def A_sets_price_optimize(self):
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
        plt.show()


    def plot_edgeworth(self, w1a, w2a, N):
        par = self.par

        # 1 Create x1A and x2A vectors
        N = N
        x1A_vec = np.linspace(0,1,N)
        x2A_vec = np.linspace(0,1,N)
        
        # 2 Define utility when consuming endowment
        uA_bar = self.utility_A(w1a, w2a)
        uB_bar = self.utility_B(1-w1a, 1-w2a)

        # 3 Create pareto combinations
        kombinationer = self.paretoC(x1A_vec, x2A_vec, uA_bar, uB_bar)
        x1, x2 = zip(*kombinationer)

        # 4 Plot settings
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

        ax_A.scatter(x1,x2,marker='s',color='royalblue',label='pareto improvements', s = 10)
        ax_A.scatter(w1a, w2a ,marker='s',color='black',label='endowment', s = 50)

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

