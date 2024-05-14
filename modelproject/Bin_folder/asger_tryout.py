import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def aggregate_demand(price_level, shift):
    shock_shift = 2  
    return 10 - price_level - shock_shift + shift  


def original_short_run_supply(price_level):
    return price_level - 4

def new_short_run_supply(price_level):
    shock_shift = 2 
    return price_level - 4 + shock_shift


def long_run_supply(output_level):
    return np.full_like(output_level, 6) 


def intersection_equation(price_level):
    return aggregate_demand(price_level, 0) - original_short_run_supply(price_level)


equilibrium_price = fsolve(intersection_equation, 5)[0]


lras_equilibrium_output = aggregate_demand(equilibrium_price, 0)


def new_intersection_equation(price_level):
    return aggregate_demand(price_level, 0) - new_short_run_supply(price_level)

new_equilibrium_price = fsolve(new_intersection_equation, 5)[0]

new_equilibrium_output = aggregate_demand(new_equilibrium_price, 0)

price_levels = np.linspace(0, 10, 100)

ad_shift = 1  
shifted_ad_curve = aggregate_demand(price_levels, ad_shift)

plt.plot(price_levels, aggregate_demand(price_levels, 0), label='Original AD')

plt.plot(price_levels, shifted_ad_curve, label='Shifted AD (After Shock)', linestyle='--', color='purple')

plt.plot(price_levels, original_short_run_supply(price_levels), label='Original SRAS')

plt.plot(price_levels, new_short_run_supply(price_levels), label='New SRAS (After Shock)', linestyle='--', color='orange')

plt.axvline(x=equilibrium_price, color='blue', linestyle='-', label='LRAS')

plt.scatter(equilibrium_price, lras_equilibrium_output, color='blue', label='Original Equilibrium')

plt.scatter(new_equilibrium_price, new_equilibrium_output, color='red', label='New Equilibrium After Shock')

def shifted_intersection_equation(price_level):
    return aggregate_demand(price_level, ad_shift) - new_short_run_supply(price_level)

new_intersection_price = fsolve(shifted_intersection_equation, 5)[0]
new_intersection_output = aggregate_demand(new_intersection_price, ad_shift)

plt.scatter(new_intersection_price, new_intersection_output, color='green', label='New Intersection Point')

plt.xlabel('Price Level')
plt.ylabel('Output Level')
plt.title('ADAS Model with Supply Shock and Adjustment Process')
plt.legend()

plt.grid(True)
plt.show()
