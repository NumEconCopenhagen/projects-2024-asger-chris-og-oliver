import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class Dataproject:
    
    def beta_table(results):
    # For this function to work, we have to have a 'results' table as an input, containing key-value pairs, where the key is the name of the object and each value is a regression result object, such as beta or a t-statistic.
    
    # Dictionary to save beta values
        beta_values = {}

        for key, result in results.items():
            beta_values[key] = round(result.params['Mkt-RF'], 3)

    # Dictionary to save constant values
        constant_values = {}

        for key, result in results.items():
            constant_values[key] = round(result.params['const'], 3)

    # Calculate t-values for the 'Mkt-RF' coefficient
        t_values = {}
        for key, result in results.items():
            t_values[key] = round(result.params['Mkt-RF'] / result.bse['Mkt-RF'], 2)
    # Calculate t-values for the constant coefficient

        t_values_a = {}
        for key, result in results.items():
            t_values_a[key] = round(result.params['const'] / result.bse['const'], 2)    
        #The function starts out with only the column with the stock name, and the beta. Then we add the other columns afterwards.
        beta_table = pd.DataFrame(list(beta_values.items()), columns=['Stock', 'Beta'])
    # Add constant values and t-values to the DataFrame
        beta_table['t_Beta'] = t_values.values()
        beta_table['cons'] = constant_values.values()
        beta_table['t_cons'] = t_values_a.values()

        return beta_table

    def mvp_w(portfolio):
        '''Calculates the weights for the minimum variance portfolio'''

        # a. define function for volatility to minimize as a function of w1 and w2
        def value_of_choice(x):
            return portfolio(x[0], x[1])[1] 
        
        # b. define bounds and restrictions
        bounds = [(-2, 2), (-2, 2)]

        # c. intitial guess and call optimizer
        guess = [0, 0]
        result = optimize.minimize(value_of_choice, guess, bounds = bounds, method='SLSQP')

        w1_min = result.x[0]
        w2_min = result.x[1]
        min_vol = result.fun

        return w1_min, w2_min, min_vol
    
    def plot_variance(portfolio, mvp_w, delta=0.5):
        '''Plots variance as a function of the first weight, white holding w2 semi-fixed'''

        # Generate weight values
        w_values = np.linspace(-0.344, 1.25, 100)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Empty volatility values
        volatility_values = []
        volatility_values2 = []
        volatility_values3 = []

        # Saving minimum variance portfolio weights and volatility
        w1_min, w2_min, min_vol = mvp_w(portfolio)


        # Change in w2_min
        delta = delta

        # Save volatility values for different weights
        for w in w_values:
            # Calculate the function values
            volatility_values.append(portfolio(w, w2_min)[1])
            volatility_values2.append(portfolio(w, w2_min - delta)[1])
            volatility_values3.append(portfolio(w, w2_min + delta)[1])

        # Plot functions value and optimum
        ax.plot(w_values, volatility_values, label = "volatility, w2_min")
        ax.plot(w_values, volatility_values2, label = "volatility, w2_min - delta")
        ax.plot(w_values, volatility_values3, label = "volatility, w2_min + delta")
        ax.scatter(w1_min, min_vol, c = "red", label = "minimum variance portfolio")


        # Add labels and a legend
        ax.set_xlabel('w1')
        ax.set_ylabel('Standard deviation')
        ax.legend(loc = "upper right")

        ax.set_xlim(-0.344, 1.25)
        ax.set_ylim(5, 8)
        plt.show()    

    def etp_w(sharpe_ratio, rf=0):
        '''Calculates weights for efficient tangent portfolio'''

        # a. define Sharpe ratio function to maximize as a function of w1 and w2
        def value_of_choice(x):
            return -sharpe_ratio(x[0], x[1], rf = rf) 
        
        # b. define bounds and restrictions
        bounds = [(-2, 2), (-2, 2)]

        # c. intitial guess and call optimizer
        guess = [0, 0]
        result = optimize.minimize(value_of_choice, guess, bounds = bounds, method='SLSQP')

        w1_etp = result.x[0]
        w2_etp = result.x[1]
        etp_sharp = -result.fun

        return w1_etp, w2_etp, etp_sharp        
    
    def plot_sharp(sharpe_ratio, etp_w, rf, delta=0.5):
        '''Plots Sharpe ratios and the efficient tangent portfolio'''

        # Generate the values for w1
        w1_values = np.linspace(-1.25, 0.5, 100)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the values of the function for each w1 value
        function_values = []
        function_values2 = []
        function_values3 = []

        # Saving the ETP weights and Sharpe ratio
        w1_etp, w2_etp, etp_sharp = etp_w(sharpe_ratio, rf=rf)

        # Change in w2_etp
        delta = delta

        # Loop through weights and save function values
        for w1 in w1_values:
            function_values.append(sharpe_ratio(w1, w2_etp, rf = rf))
            function_values2.append(sharpe_ratio(w1, w2_etp - delta, rf = rf)) 
            function_values3.append(sharpe_ratio(w1, w2_etp + delta, rf = rf))  

        # Plot Sharpe ratios and the optimum
        ax.plot(w1_values, function_values, label = "sharpe ratio, w2_etp")
        ax.plot(w1_values, function_values2, label = "sharpe ratio, w2_etp - delta")
        ax.plot(w1_values, function_values3, label = "sharpe ratio, w2_etp + delta")
        ax.scatter(w1_etp, etp_sharp, c = "red", label = "efficient tangent portfolio")

        # Add labels and a legend
        ax.set_xlabel('w1')
        ax.set_ylabel('Sharpe ratio')
        ax.legend(loc = "lower right")

        ax.set_xlim(-1.25, 0.5)
        ax.set_ylim(0.3, .62)
        plt.show()

    def plot_capm(mvp, etp, rf, lin_vec, portfolio, return_matrix, volatility_matrix, data):
        '''Plots the efficient frontier and the capital market line'''
        fig, ax = plt.subplots()

        # Empty vector to store frontier
        frontier = []

        for l in lin_vec:
            # Calculate the function values
            frontier.append(portfolio(l[0], l[1])) 

        # Unpacking values in frontier to plot against each other    
        return_values, vol_values = zip(*frontier)

        # Points for risk free rate and ETP
        rf = rf
        point_rf = [0, rf]
        point_etp = [etp[1], etp[0]]

        # Compute the equation of the line passing through the two points
        # y = bx + c, where m is the slope and c is the y-intercept
        slope = (point_etp[1] - point_rf[1]) / (point_etp[0] - point_rf[0])
        y_intercept = point_rf[1] - slope * point_rf[0]

        # Generate x values for the line
        x_values = np.linspace(0, 13, 100)

        # Compute corresponding y values
        y_values = slope * x_values + y_intercept

        # Plot the capital market line i.e. tangent line
        plt.plot(x_values, y_values, label='Capital market line', linewidth = 0.5, color = "black")

        # Plot Efficient frontier and different portfolios and stocks
        ax.plot(vol_values, return_values, label = "efficient frontier", linewidth = 0.5, color = "blue")
        ax.scatter(mvp[1], mvp[0], c = "green", label = "minimum variance portfolio")
        ax.scatter(etp[1], etp[0], c = "red", label = "efficient tangent portfolio")
        ax.scatter(0, rf, c = "black", marker = "o", label = "Risk Free return")

        # Plot stocks on the graph
        volatility_values = volatility_matrix.values
        return_values = return_matrix.values
        ax.scatter(volatility_values, return_values, c = ["purple", "blue", "black"])

        # Annotate text next to the stocks and MVP, ETP points
        for i, txt in enumerate(data.columns[0:3]):
            ax.annotate(txt, (volatility_values[i], return_values[i]), xytext=(5, 5), textcoords='offset points')
        ax.annotate("MVP", (mvp[1], mvp[0]), xytext=(-38, -3), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
        ax.annotate("ETP", (etp[1], etp[0]), xytext=(-38, -3), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

        # Add labels and a legend
        ax.set_xlabel('volatility')
        ax.set_ylabel('return')
        ax.legend()

        ax.set_xlim(0, 25)
        ax.set_ylim(0, 7)
        plt.show()