import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mvp_w(N, portfolio):
    '''Calculates the minimum variance portfolio for given return and covariance matrix'''

    # Create linear weight vector
    N = N
    w_vec = np.linspace(-1, 2, N)

    # Initial best volatility and weight
    vol_best = np.inf
    w_best = 0

    # Loops over all weight and saves the weight if the volatility is lower than before
    for w in w_vec:
        vol = portfolio(w)[1] # Calculates volatility for this portfolio
        if vol < vol_best:
            vol_best = vol
            w_best = w
    return w_best

def plot_variance(portfolio):
    '''Plots the variance as well as the minimum variance portfolio'''
    # Generate weight values
    w_values = np.linspace(-1, 2, 100)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Empty volatility values
    volatility_values = []

    for w in w_values:
        # Calculate the function values
        volatility_values.append(portfolio(w)[1])
        
    # Calculate minimum variance portfolio
    w_min = mvp_w(100, portfolio)
    mvp = portfolio(w_min)

    # Plot functions value and optimum
    ax.plot(w_values, volatility_values, label = "volatility")
    ax.scatter(w_min, mvp[1], c = "red", label = "minimum variance portfolio")

    # Add labels and a legend
    ax.set_xlabel('weight')
    ax.set_ylabel('Standard deviation')
    ax.legend()

    ax.set_xlim(-1, 2)
    plt.show()

def sharpe_ratio(w, portfolio, rf = 0):
    '''Calculate the sharpe ratio for any portfolio'''

    rf = rf # Risk free rate
    e_r, sd = portfolio(w) # Expected return and volatility

    sharpe = (e_r - rf)/sd
    return sharpe

def etp_w(N, portfolio):
    '''Calculates Efficient Tangent Portfolio by finding the highest Sharpe ratio'''

    # Create linear weight vector
    N = N
    w_vec = np.linspace(-1, 2, N)

    # Initial best volatility and weight
    sharpe_best = -np.inf
    w_best = 0

    # Loops over all weight and saves the weight if the volatility is lower than before
    for w in w_vec:
        sharpe = sharpe_ratio(w, portfolio)
        if sharpe > sharpe_best:
            sharpe_best = sharpe
            w_best = w
    return w_best

def plot_sharp(portfolio):
    '''Plots Sharpe ratios and the efficient tangent portfolio'''

    # Generate the values for x, y, and z
    w_values = np.linspace(-1, 2, 100)  # Varying values for the third input

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the values of the function for each z value
    function_values = []

    for w in w_values:
        # Calculate the function values
        function_values.append(sharpe_ratio(w, portfolio)) 
    
    # Calculate ETP
    w_etp = etp_w(100, portfolio)

    ax.plot(w_values, function_values, label = "sharpe ratio")
    ax.scatter(w_etp, sharpe_ratio(w_etp, portfolio), c = "red", label = "efficient tangent portfolio")

    # Add labels and a legend
    ax.set_xlabel('weight')
    ax.set_ylabel('Sharpe ratio')
    ax.legend()

    ax.set_xlim(-1, 2)
    plt.show()

def plot_capm(mvp, etp, lin_vec, portfolio):
    '''Plots the efficient frontier and the capital market line'''
    fig, ax = plt.subplots()

    # Empty vector to store frontier
    frontier = []

    for l in lin_vec:
        # Calculate the function values
        frontier.append(portfolio(l)) 
        
    return_values, vol_values = zip(*frontier)

    rf = 0
    point_rf = [0, rf]
    point_etp = [etp[1], etp[0]]

    # Compute the equation of the line passing through the two points
    # y = mx + c, where m is the slope and c is the y-intercept
    slope = (point_etp[1] - point_rf[1]) / (point_etp[0] - point_rf[0])
    y_intercept = point_rf[1] - slope * point_rf[0]

    # Generate x values for the line
    x_values = np.linspace(0, 10, 100)
    # Compute corresponding y values
    y_values = slope * x_values + y_intercept

    # Plot the points and the line passing through them
    plt.plot(x_values, y_values, label='Capital market line', linewidth = 0.5, color = "black")


    ax.plot(vol_values, return_values, label = "efficient frontier")
    #ax.scatter(0, rf, label = "risk free")
    ax.scatter(mvp[1], mvp[0], c = "green", label = "minimum variance portfolio")
    ax.scatter(etp[1], etp[0], c = "red", label = "efficient tangent portfolio")

    # Add labels and a legend
    ax.set_xlabel('volatility')
    ax.set_ylabel('return')
    ax.legend()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    plt.show()
