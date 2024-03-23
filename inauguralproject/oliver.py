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
        par.w1B = 1-par.w1A
        par.w2B = 1-par.w2A

    def utility_A(self,x1A,x2A):
        par = self.par
        if x1A < 0 or x2A < 0:
            util_A = 0
        else:
            util_A = x1A**(par.alpha)*x2A**(1-par.alpha)
        print(util_A)
        return util_A
    
    def utility_B(self,x1B,x2B):
        par = self.par
        if x1B < 0 or x2B < 0:
            util_B = 0
        else:
            util_B = x1B**(par.beta)*x2B**(1-par.beta)
        print(util_B)
        return util_B
    
    def demand_A(self,p1):
        par = self.par
        I_A = p1*par.w1A+1*par.w2A
        x1A_star = par.alpha * (I_A/p1)
        x2A_star = (1-par.alpha)*(I_A/1)
        print(x1A_star,x2A_star)
        return x1A_star, x2A_star
        
    def demand_B(self,p1):
        par = self.par
        I_B = p1*par.w1B+1*par.w2B
        x1B_star = par.beta * (I_B/p1)
        x2B_star = (1-par.beta) * (I_B/1)
        print(x1B_star, x2B_star)
        return x1B_star, x2B_star

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    #Definerer en funktion, der returnerer settet C, der illustrerer alle de mulige kombinationer af x_1^A og x_2^A, 
    #der giver begge forbrugere mindst så meget nytte som ved deres initielle endowment  w1A og w2A.
    # Først defineres x_1^A og x_2^A vektorerne, from (0,1/75,2/75,...,74/75,1).
    # Then, we define uA_bar and uB_bar, which is the utility of, respectively, consumer A and consumer B, given their initial endowments.

    # Then, we create the empty list "kombinationer", for which we iterate over possible combinations for x1A and x2A, starting with x1A,x2A=[0,0], returning their utility values.
    # Filling out the list, we achieve utilities for uA and uB (so there are two lists in the list). For a given pair of x1a,x2a,x1b,x2b, if the combination is an pareto improvement, that is
    # the combination yields the same or a higher utility than the initial endowment, the combination is added to the list.
    def paretoC(self, N = 75):
        N = N
        x1A_vec = np.linspace(0,1,N)
        x2A_vec = np.linspace(0,1,N)
        uA_bar = self.utility_A(self.par.w1A, self.par.w2A)
        uB_bar = self.utility_B(self.par.w1B, self.par.w2B)

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
    
    def P_1(self, N = 75):
        N = N
        P_1 = [0.5]
        i = 1
        while P_1[-1] < 2.5:
            P_1.append(0.5 + (2*i)/N)
            i += 1 
        return P_1


    def plot_edgeworth(self, N = 75):
        par = self.par

        # Create pareto combinations
        kombinationer = self.paretoC(N)
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