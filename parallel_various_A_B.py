import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import math

# user input demand, yard data
k = 50
epsilon = 20              # range of capacity excess
capacity = 100000          # initial capacity
A = 1000000               # initial solution: yard length (ft)
B = 1000000            # initial solution: yard width (ft)

# model iterations
iteration_count = 0
max_iterations = 2000


# Record the objective values for each iteration
decision_variables = []
iteration_values = []
results = []


# parallel layout input parameters
m = 200                 # railcars per train
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
S_p = 400               # the area of a standard parking spot (40 ft * 10 ft)
P = 30                  # the width of each aisle
BL_w = 80               # the width of each parking block


def d_t_y(M, N, n_p, n_r, n_t):
    return 10 * n_r * M + 80*N + (M + N + 1.5) * n_p * P + 2 * n_t * P


def d_g(M, N, n_p, n_r):
    return 10 * n_r * M + 80 * N + (M + N + 1)* n_p * P


def ugly_sigma(x):
    total_sum = 0
    for i in range(1, x):
        total_sum += 2 * i * (x - i)
    result = total_sum / (x ** 2)
    return result


def d_r(M, N, n_r,n_p):
    return 5 * n_r +40 + ugly_sigma(M) * (10 * n_r + n_p*P) + ugly_sigma(N) * (80 + n_p * P)


def objective(x):
    M, N, n_t, n_p, n_r = x
    M, N, n_t, n_p, n_r = math.ceil(M), math.ceil(N), math.ceil(n_t), math.ceil(n_p), math.ceil(n_r)
    efficiency_obj = d_t_y(M, N, n_p, n_r, n_t) + d_g(M, N, n_p, n_r) + d_r(M, N, n_r, n_p)
    return efficiency_obj


def check_constraints(params):
    global A, B, capacity, k
    M, N, n_t, n_p, n_r = params
    capacity = math.ceil(M) * math.ceil(N) * 2 * math.ceil(n_r)
    A = (math.ceil(M) + 1) * math.ceil(n_p) * P + math.ceil(M) * 10 * math.ceil(n_r)
    B = (math.ceil( N) + 1) * math.ceil(n_p) * P + math.ceil(N) * BL_w
    capacity_diff = capacity - k
    # Check: planned capacity is close enough to expected capacity k ?
    if capacity > k and capacity_diff <= epsilon:
        decision_variables.append([math.ceil(M), math.ceil(N), math.ceil(n_r)])
        # print(f"When k = {k}, Capacity is {capacity} vs. k is {k}, and capacity_diff is {capacity_diff}. There is no penalty.")
        return 0
    else:
        penalty = np.infty
        return penalty


def constrained_objective(x):
    penalty = check_constraints(x)
    return objective(x) + penalty  # If not satisfied, return objective with penalty


def callback_function(x, convergence):
    global iteration_count
    iteration_count += 1
    if iteration_count >= max_iterations:
        print(f"Warning: Maximum iteration limit reached ({max_iterations}).")
    else:
        current_value = objective(x)
        iteration_values.append(current_value)


# Decision variable
bounds = [(1, 10000),      # M: The number of columns of parking blocks in the layout
          (1, 10000),      # N: The number of rows of parking blocks in the layout
          (2, 4),          # n_t: numbers of from the train side
          (2, 4),          # n_p: numbers of from the train side
          (10, 10000)]     # n_r: The number of each row spots within every parking block (10 * n_r = BL_l, the length of each parking block)

# try different k
try:
    k_max = A * B / S_p
    for k in range(3000, 8005, 100):
        if k >= k_max:
            print(f"Error: The expected demand is larger than the actual maximum spot {int(k_max)}.")
            break

        result_de = differential_evolution(constrained_objective, bounds, strategy='best1bin',
                                           maxiter=1000, tol=1e-7, mutation=(0.5, 1), recombination=0.7)

        result_de.x = list(map(int, result_de.x))
        M = decision_variables[-1][0]
        N = decision_variables[-1][1]
        n_t = result_de.x[2]
        n_p = result_de.x[3]
        n_r = decision_variables[-1][2]
        value = result_de.fun
        capacity = M * N * 2 * n_r
        length = (M + 1) * n_p * P + M * 10 * n_r
        width = (N + 1) * n_p * P + N * BL_w
        results.append([k, M, N, n_t, n_p, n_r, value, capacity, length, width])
        print(f"Processed k = {k}, where the row data is {[k, M, N, n_t, n_p, n_r, value, capacity, length, width]}")

# Remind error occurs
except Exception as e:
    print(f"Last call ends in k = {k}, Error: {e}")

df = pd.DataFrame(results, columns=["k", "M", "N", "n_t", "n_p", "n_r", "value", "capacity_check", "A", "B"])
df.to_excel("k_results_various_A_B_new.xlsx", index=False)
print("Done!")

# # Plot the convergence curve
# plt.plot(iteration_values, label="Total distances")
# plt.xlabel("Iteration")
# plt.ylabel("Objective Value")
# plt.title("Convergence of Differential Evolution Method for the Various Yard Case")
# plt.legend()
# plt.grid(True)
# plt.show()