import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# parallel layout input parameters
A = 1928    # yard length (ft)
B = 765     # yard width (ft)
b = 30   # efficiency constraint of d_t
c = 30   # efficiency constraint of d_y
d = 30   # efficiency constraint of d_g
f = 1459200    # land constraint, which is smaller than 1928*765
k = 150   # loaded train units, from the timetable
l = 60  # each railcar length (60 ft, around 18 meters)
s = 72  # train-yard distance (ft)
v_hostler = 1320  # hostler speed (15mph = 79200 ft/hr = 1320 ft/min)
v_truck = 1320    # truck speed (15 mph)
delta_k = np.ones(k)    # assume fully loaded, which could be binary
n_k = 1     # container on each train: assume fully loaded with 1 container
c_p = 1     # tiers of containers
S_p = 400    # the area of a standard parking spot (ft^2)， from 19 feet long and 8 feet  --> 40 * 10 as a parking spot
P = 30      # The width of each lane
Q = 160     # The length of each aisle
BL_w = 80    # The width of each parking block

# # decision variable
# M = 7       # The number of columns of parking blocks in the layout
# N = 3       # The number of rows of parking blocks in the layout
# n_t = 2     # numbers of from the train side
# n_p = 2     # numbers of from the train side
# BL_l = 15   # The length of each parking block  (Decision variable)


def d_t(n_t):
    return 2 * (k * l + s - B) + 2 * n_t * P


def d_y(M, N, n_p, BL_l):
    return 2 * (0.5 * n_p * P + 0.5 * BL_l) + 2 * (1.5 * n_p * P + BL_w)


def d_g(M, N, n_p):
    return 2 * n_p * P


def objective(x):
    M, N, n_t, n_p, BL_l = x
    M, N, n_t, n_p = int(M), int(N), int(n_t), int(n_p)

    return (d_t(n_t) / v_hostler
            + d_y(M, N, n_p, BL_l) / v_hostler
            + d_g(M, N, n_p) / v_truck)


def penalty(x):
    M, N, n_t, n_p, BL_l = x
    M, N, n_t, n_p = int(M), int(N), int(n_t), int(n_p)

    penalties = []
    penalties.append(max(0, d_t(n_t) / v_hostler - b))
    penalties.append(max(0, d_y(M, N, n_p, BL_l) / v_hostler - c))
    penalties.append(max(0, d_g(M, N, n_p) / v_truck - d))
    penalties.append(max(0, n_t * P * Q - f))
    penalties.append(max(0, (c_p * M * N * BL_w * BL_l / S_p) - np.sum(delta_k * n_k)))

    N_P_constraints = [0, (N + 1) * P * n_p + N * BL_l - B]
    M_Q_constraints = [0, (M + 1) * P * n_p + M * BL_w - A]

    penalties.extend([max(0, -c) for c in N_P_constraints])
    penalties.extend([max(0, -c) for c in M_Q_constraints])

    return np.sum(penalties)


def combined_objective(x):
    return objective(x) + penalty(x)


bounds = [(1, 20),  # M: The number of rows of parking blocks in the layout
          (1, 20),  # N: The number of columns of parking blocks in the layout
          (2, 5),  # n_t:
          (2, 5),  # n_p
          (10, 800)]  # BL_l

# Record the objective values for each iteration
iteration_values = []


def callback(x, convergence):
    iteration_values.append(combined_objective(x))


result_de = differential_evolution(combined_objective, bounds, strategy='best1bin',
                                   maxiter=1000, tol=1e-7, mutation=(0.5, 1), recombination=0.7)


# Ensure the final solution is integer
result_de.x = list(map(int, result_de.x))

print("Differential Evolution result:")
print(result_de)


# # Plot the convergence curve
# plt.plot(iteration_values, label="Objective Value")
# plt.xlabel("Iteration")
# plt.ylabel("Objective Value")
# plt.title("Convergence of Differential Evolution")
# plt.legend()
# plt.grid(True)
# plt.show()