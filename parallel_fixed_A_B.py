import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
import math

# Record the decision variables and objective values for each iteration
decision_variables = []
objective_values = []
results = []

# user input demand, yard data
k = 1000                 # expected demand (TEU)
A = 1928                # yard length (ft)
B = 765                 # yard width (ft)

# parallel layout input parameters
b = 30                  # efficiency constraint of d_t
c = 30                  # efficiency constraint of d_y
d = 30                  # efficiency constraint of d_g
f = 1459200             # land constraint, which is smaller than 1928*765
l = 60                  # each railcar length (60 ft, around 18 meters)
s = 72                  # train-yard distance (ft)
v_hostler = 1320        # hostler speed (15mph = 79200 ft/hr = 1320 ft/min)
v_truck = 1320          # truck speed (15 mph)
delta_k = np.ones(k)    # binary loading ratio, where we assume all railcars are fully loaded
n_k = 1                 # container on each train: assume fully loaded with 1 container
c_p = 1                 # tiers of containers
S_p = 400               # the area of a standard parking spot (ft^2), which is obtained from 19 feet long and 8 feet (40 * 10 as a parking spot)
P = 30                  # the width of each lane
Q = 160                 # the length of each aisle
BL_w = 80               # the width of each parking block
k_max = A*B/S_p         # the maximum parking spots the actual yard can provide


def d_t(N, n_t, n_p, n_r):
    return k * l + s - 0.5 * ((N + 1) * P * n_p + N * 10 * n_r) + n_t * P    # avg


def d_y(M, N, n_p, n_r):
    return (N + 1)**2 / (2 * N**2) * (A + ((N + 1) * P * n_p + N * 10 * n_r)) + 2 * (M + 2) * n_p * P    # avg


def d_g(M, N, n_p, n_r):
    return (N + 2)/(2 * N) + (N + 1)**2 / (2 * N**2) * (A + ((N + 1) * P * n_p + N * 10 * n_r)) + 2 * (M + 1) * n_p * P # avg


def check_constraints(params):
    M, N, n_t, n_p, n_r = params
    print("M", math.ceil(M))
    print("N", math.ceil(N))
    print("n_r", math.ceil(n_r))
    capacity = math.ceil(M) * math.ceil(N) * 2 * math.ceil(n_r)
    length = (math.ceil(M) + 1) * math.ceil(n_p) * P + math.ceil(M) * 10 * math.ceil(n_r)
    width = (math.ceil(N) + 1) * math.ceil(n_p) * P + math.ceil(N) * BL_w
    if capacity > k and length <= A and width <= B:
        M = math.ceil(M)
        N = math.ceil(N)
        n_r = math.ceil(n_r)
        decision_variables.append([math.ceil(M), math.ceil(N), math.ceil(n_r)])
        return True
    return False


def objective(x):
    M, N, n_t, n_p, n_r = x
    M, N, n_t, n_p, n_r = math.ceil(M), math.ceil(N), math.ceil(n_t), math.ceil(n_p), math.ceil(n_r)
    efficiency_obj = d_t(N, n_t, n_p, n_r) + d_y(M, N, n_p, n_r) + d_g(M, N, n_p, n_r)
    return efficiency_obj


def constrained_objective(x):
    if check_constraints(x):
        return objective(x)  # if satisfied, return objective
    else:
        return np.infty # if not satisfy, return infinity


def callback_function(x, convergence):
    current_value = objective(x)
    objective_values.append(current_value)


# # Decision variable
# M: The number of columns of parking blocks in the layout
# N: The number of rows of parking blocks in the layout
# n_t: numbers of from the train side
# n_p: numbers of from the train side
# n_r: The number of spots within each parking block (10 * n_r = BL_l, the length of each parking block)


bounds = [(1, 50),      # M
          (1, 50),      # N
          (2, 4),       # n_t
          (2, 4),       # n_p
          (1, 50)]     # n_r


if k >= k_max:
    print(f"Error: The expected demand is larger than the actual maximum spot {int(k_max)}.")
else:
    # penalty-based differential evolution algorithm
    result_de = differential_evolution(constrained_objective, bounds, strategy='best1bin',
                                       maxiter=1000, tol=1e-7, mutation=(0.5, 1), recombination=0.7, callback=callback_function)
    # Ensure the final solution is integer
    result_de.x = list(map(int, result_de.x))
    print("Differential Evolution result:")
    print(result_de)
    print("decision_variables:", decision_variables[-1])
    M = decision_variables[-1][0]
    N = decision_variables[-1][1]
    n_t = result_de.x[2]
    n_p = result_de.x[3]
    n_r = decision_variables[-1][2]
    value = result_de.fun
    capacity = M * N * 2 * n_r
    length = (M+1) * n_p * P + M * 10 * n_r
    width = (N+1) * n_p * P + N * BL_w
    row = [M, N, n_t, n_p, n_r, value, capacity, length, width]
    print(f"The decision making for the case k = {k} is: {row}")

# Plot the convergence curve
plt.plot(objective_values, label="Total distances")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence of Differential Evolution")
plt.legend()
plt.grid(True)
plt.show()


# # try different k
# try:
#     for k in range(790, 1001, 5):
#         if k >= k_max:
#             print(f"Error: The expected demand is larger than the actual maximum spot {int(k_max)}.")
#             break
#
#         result_de = differential_evolution(constrained_objective, bounds, strategy='best1bin',
#                                            maxiter=1000, tol=1e-7, mutation=(0.5, 1), recombination=0.7)
#
#         result_de.x = list(map(int, result_de.x))
#         M = decision_variables[-1][0]
#         N = decision_variables[-1][1]
#         n_t = result_de.x[2]
#         n_p = result_de.x[3]
#         n_r = decision_variables[-1][2]
#         value = result_de.fun
#         capacity = M * N * 2 * n_r
#         length = (M+1) * n_p * P + M * 10 * n_r
#         width = (N+1) * n_p * P + N * BL_w
#         results.append([k, M, N, n_t, n_p, n_r, value, capacity, length, width])
#         print(f"Processed k = {k}, where the row data is {[k, M, N, n_t, n_p, n_r, value, capacity, length, width]}")
#
# # Record k if error occor
# except Exception as e:
#     print(f"Last call ends in k = {k}, Error: {e}")
#
# df = pd.DataFrame(results, columns=["k", "M", "N", "n_t", "n_p", "n_r", "value", "capacity_check", "length_check", "width_check"])
# df.to_excel("k_results.xlsx", index=False)
# print("Done!")