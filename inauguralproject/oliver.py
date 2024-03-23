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