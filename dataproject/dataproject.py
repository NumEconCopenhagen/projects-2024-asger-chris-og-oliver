import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataproject:
    
    def beta_table(results):
    
    # List to save beta values
        beta_values = {}

        for key, result in results.items():
            beta_values[key] = round(result.params['Mkt-RF'], 3)

    # Extract constant values
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

        beta_table = pd.DataFrame(list(beta_values.items()), columns=['Stock', 'Beta'])

    # Add constant values and t-values to the DataFrame
        beta_table['t_Beta'] = t_values.values()
        beta_table['cons'] = constant_values.values()
        beta_table['t_cons'] = t_values_a.values()

        return beta_table

