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
    
    # We define a function P_1 used in Q2 on market errors. The function starts in
    # P_1 = =0,5 and i = 1. The while loop stops when the last value of P_1 is below 2.5.
    # As long as the last value P_1<2.5, we add an observation with the value 0.5+(2*i)/n, to the list P_1.
    # then, we go onto the next value of i.
    # The function simply just returns a list P_1.
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


# Definerer en funktion, market_clear, der for hver værdi i P_1 returnerer værdierne eps1 og eps2 (fra check_market_clearing funktionen) og navngiver
# dem e1_now, e2_now. Hvis den absolutte værdi af fejlene er mindre end 10 (why 10?), så sætter vi en e1 = e1_now.
# og e1_best og e2_best = e1_now og e2_now. Og så p1_best = den p1 værdi vi er i gang med.
# til sidst returnerer vi errorsne og prisen.
# dette loop fortsætter vi med for alle 75 observationer i p1, og vi prøver dermed at finde den pris p1_best der giver de allermindst errors e1_best og e2_best.
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

    def plot_error(self, p1, N = 75):
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