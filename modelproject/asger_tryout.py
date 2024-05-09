import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the Aggregate Demand (AD) curve
def aggregate_demand(price_level, shift):
    # Introducing a positive demand shock by shifting the AD curve upward
    shock_shift = 2  # arbitrary shift to demonstrate the shock
    return 10 - price_level - shock_shift + shift  # Subtracting the shock_shift to shift the curve to the right

# Define the original Short-Run Aggregate Supply (SRAS) curve
def original_short_run_supply(price_level):
    return price_level - 4

# Define the new Short-Run Aggregate Supply (SRAS) curve after the shock
def new_short_run_supply(price_level):
    # Introducing a negative supply shock by shifting the SRAS curve upward
    shock_shift = 2  # arbitrary shift to demonstrate the shock
    return price_level - 4 + shock_shift

# Define the Long-Run Aggregate Supply (LRAS) curve
def long_run_supply(output_level):
    return np.full_like(output_level, 6)  # Create an array with the same shape as output_level, filled with the LRAS equilibrium price level

# Define the equation for finding the intersection of AD and SRAS curves
def intersection_equation(price_level):
    return aggregate_demand(price_level, 0) - original_short_run_supply(price_level)

# Find the equilibrium price level using fsolve
equilibrium_price = fsolve(intersection_equation, 5)[0]

# Calculate the LRAS equilibrium output level
lras_equilibrium_output = aggregate_demand(equilibrium_price, 0)

# Define the equation for finding the intersection of New SRAS and Original AD curves
def new_intersection_equation(price_level):
    return aggregate_demand(price_level, 0) - new_short_run_supply(price_level)

# Find the new equilibrium price level using fsolve
new_equilibrium_price = fsolve(new_intersection_equation, 5)[0]

# Calculate new equilibrium output level
new_equilibrium_output = aggregate_demand(new_equilibrium_price, 0)

# Generate price levels
price_levels = np.linspace(0, 10, 100)

# Calculate the effect of the shock on the AD curve
ad_shift = 1  # arbitrary shift to demonstrate the effect of the shock
shifted_ad_curve = aggregate_demand(price_levels, ad_shift)

# Plot the original Aggregate Demand (AD) curve
plt.plot(price_levels, aggregate_demand(price_levels, 0), label='Original AD')

# Plot the shifted Aggregate Demand (AD) curve after the shock
plt.plot(price_levels, shifted_ad_curve, label='Shifted AD (After Shock)', linestyle='--', color='purple')

# Plot the original Short-Run Aggregate Supply (SRAS) curve
plt.plot(price_levels, original_short_run_supply(price_levels), label='Original SRAS')

# Plot the new Short-Run Aggregate Supply (SRAS) curve after the shock
plt.plot(price_levels, new_short_run_supply(price_levels), label='New SRAS (After Shock)', linestyle='--', color='orange')

# Plot the Long-Run Aggregate Supply (LRAS) line vertically at the equilibrium point
plt.axvline(x=equilibrium_price, color='blue', linestyle='-', label='LRAS')

# Plot the original equilibrium point
plt.scatter(equilibrium_price, lras_equilibrium_output, color='blue', label='Original Equilibrium')

# Plot the new equilibrium point after the shock
plt.scatter(new_equilibrium_price, new_equilibrium_output, color='red', label='New Equilibrium After Shock')

# Define the equation for finding the intersection of Shifted AD and Shifted AS curves
def shifted_intersection_equation(price_level):
    return aggregate_demand(price_level, ad_shift) - new_short_run_supply(price_level)

# Find the new intersection point using fsolve
new_intersection_price = fsolve(shifted_intersection_equation, 5)[0]
new_intersection_output = aggregate_demand(new_intersection_price, ad_shift)

# Plot the new intersection point between Shifted AD and Shifted AS curves
plt.scatter(new_intersection_price, new_intersection_output, color='green', label='New Intersection Point')

# Add labels and title
plt.xlabel('Price Level')
plt.ylabel('Output Level')
plt.title('ADAS Model with Supply Shock and Adjustment Process')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
