import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt  

def find_ABCD(X, y):

    # Objective function to measure point distance
    dist = lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    # Finding A
    # Initial optimal values
    dist_min_A = np.inf
    x1_A = np.nan
    x2_A = np.nan

    for x in X:
        if x[0] > y[0] and x[1] > y[1]: # Binding condition for A
            dist_now_A = dist(x) # Calculating distance
            if dist_now_A < dist_min_A: # If distance smaller than minimum, then update optimum
                dist_min_A = dist_now_A
                x1_A = x[0]
                x2_A = x[1]

    # Finding B
    dist_min_B = np.inf
    x1_B = np.nan
    x2_B = np.nan

    for x in X:
        if x[0] > y[0] and x[1] < y[1]: # Binding condition for B
            dist_now_B = dist(x) # Calculating distance
            if dist_now_B < dist_min_B: # If distance smaller than minimum, then update optimum
                dist_min_B = dist_now_B
                x1_B = x[0]
                x2_B = x[1]        

    # Finding C
    dist_min_C = np.inf
    x1_C = np.nan
    x2_C = np.nan

    for x in X:
        if x[0] < y[0] and x[1] < y[1]: # Binding condition for C
            dist_now_C = dist(x) # Calculating distance
            if dist_now_C < dist_min_C: # If distance smaller than minimum, then update optimum
                dist_min_C = dist_now_C
                x1_C = x[0]
                x2_C = x[1]

    # Finding D
    # Initial optimal values
    dist_min_D = np.inf
    x1_D = np.nan
    x2_D = np.nan

    for x in X:
        if x[0] < y[0] and x[1] > y[1]: # Binding condition for D
            dist_now_D = dist(x) # Calculating distance
            if dist_now_D < dist_min_D: # If distance smaller than minimum, then update optimum
                dist_min_D = dist_now_D
                x1_D = x[0]
                x2_D = x[1]


    ABC = [(x1_A, x1_B, x1_C), (x2_A, x2_B, x2_C)]
    CDA = [(x1_C, x1_D, x1_A), (x2_C, x2_D, x2_A)]

    return ABC, CDA


def find_nan(X, y):

    # Objective function to measure point distance
    dist = lambda x: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    # Finding A
    # Initial optimal values
    dist_min_A = np.inf
    x1_A = np.nan
    x2_A = np.nan

    for x in X:
        if x[0] > y[0] and x[1] > y[1]: # Binding condition for A
            dist_now_A = dist(x) # Calculating distance
            if dist_now_A < dist_min_A: # If distance smaller than minimum, then update optimum
                dist_min_A = dist_now_A
                x1_A = x[0]
                x2_A = x[1]

    # Finding B
    dist_min_B = np.inf
    x1_B = np.nan
    x2_B = np.nan

    for x in X:
        if x[0] > y[0] and x[1] < y[1]: # Binding condition for B
            dist_now_B = dist(x) # Calculating distance
            if dist_now_B < dist_min_B: # If distance smaller than minimum, then update optimum
                dist_min_B = dist_now_B
                x1_B = x[0]
                x2_B = x[1]        

    # Finding C
    dist_min_C = np.inf
    x1_C = np.nan
    x2_C = np.nan

    for x in X:
        if x[0] < y[0] and x[1] < y[1]: # Binding condition for C
            dist_now_C = dist(x) # Calculating distance
            if dist_now_C < dist_min_C: # If distance smaller than minimum, then update optimum
                dist_min_C = dist_now_C
                x1_C = x[0]
                x2_C = x[1]

    # Finding D
    # Initial optimal values
    dist_min_D = np.inf
    x1_D = np.nan
    x2_D = np.nan

    for x in X:
        if x[0] < y[0] and x[1] > y[1]: # Binding condition for D
            dist_now_D = dist(x) # Calculating distance
            if dist_now_D < dist_min_D: # If distance smaller than minimum, then update optimum
                dist_min_D = dist_now_D
                x1_D = x[0]
                x2_D = x[1]


    ABC = [(x1_A, x1_B, x1_C), (x2_A, x2_B, x2_C)]
    CDA = [(x1_C, x1_D, x1_A), (x2_C, x2_D, x2_A)]

    return x1_A, x2_A, x1_B, x2_B, x1_C, x2_C, x1_D, x2_D, ABC, CDA

def plt_nan(X, Y, x1_A, x2_A, x1_B, x2_B, x1_C, x2_C, x1_D, x2_D, ABC, CDA):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], color='grey', marker='o', label="X")

    plt.scatter(Y[2][0], Y[2][1], color='red', marker='o', label="Y[2]")
    plt.annotate('Y[2]', (Y[2][0], Y[2][1]), textcoords="offset points", xytext=(15, -2), ha='center')

    plt.scatter(x1_A, x2_A, color='blue', marker='x', label="A")
    plt.annotate('A', (x1_A, x2_A), textcoords="offset points", xytext=(7.5,5), ha='center')

    plt.scatter(x1_B, x2_B, color='black', marker='x', label="B")
    plt.annotate('B', (x1_B, x2_B), textcoords="offset points", xytext=(10,-10), ha='center')

    plt.scatter(x1_C, x2_C, color='purple', marker='x', label="C")
    plt.annotate('C', (x1_C, x2_C), textcoords="offset points", xytext=(-10,-10), ha='center')

    plt.scatter(x1_D, x2_D, color='red', marker='x', label="D")
    plt.annotate('D', (x1_D, x2_D), textcoords="offset points", xytext=(-7.5,5), ha='center')

    # Drawing triangles ABC and CDA
    plt.fill(ABC[0], ABC[1], color='blue', alpha=0.5)
    plt.fill(CDA[0], CDA[1], color='skyblue', alpha=0.5)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Points in unit square')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()