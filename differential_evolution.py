import numpy as np
from scipy.optimize import differential_evolution

# Given parameters
A = 44      # yard width
B = 160     # yard length
a = 8000    # constraints of land use
b = 5       # constraints of train side efficiency
c = 5       # constraints of yard side efficiency
d = 5       # constraints of gate side efficiency
f = 4000    # constraints of total aisle areas
k = 4       # The numbers of railcar units
l = 20      # The length of each railcar and the joint part
s = 10      # The vertical distance between the train head to the edge of the yard
v_hostler = 20
v_truck = 20
delta_k = np.ones(k)  # loading ratio of each railcar set to 1
n_k = 1     # numbers of containers per railcar
c_p = 1     # capacity for each parking spot
S_p = 60    # area of a standard parking spot

P = 10      # fixed aisle width
Q = 160     # fixed aisle length
BL_l = 15   # fixed block length
BL_w = 4    # fixed block width

def d_t(n_t):
    return 2 * (k * l + s - B) + 2 * n_t * P

def d_y(M, N, n_p):
    return 2 * (0.5 * n_p * P + 0.5 * BL_l) + 2 * (1.5 * n_p * P + BL_w)

def d_g(M, N, n_p):
    return 2 * n_p * P

def objective(x):
    M, N, n_t, n_p = map(int, x)
    return (d_t(n_t) / v_hostler
            + d_y(M, N, n_p) / v_hostler
            + d_g(M, N, n_p) / v_truck)

def penalty(x):
    M, N, n_t, n_p = map(int, x)
    penalties = []

    # Constraints
    penalties.append(max(0, d_t(n_t) / v_hostler - b))  # d_t constraint
    penalties.append(max(0, d_y(M, N, n_p) / v_hostler - c))  # d_y constraint
    penalties.append(max(0, d_g(M, N, n_p) / v_truck - d))  # d_g constraint
    penalties.append(max(0, n_t * P * Q - f))  # n_tPQ constraint
    penalties.append(max(0, (c_p * M * N * BL_w * BL_l / S_p) - np.sum(delta_k * n_k)))  # c_p constraint

    N_P_constraints = [0.5 * B - N * P * n_p, N * P * n_p - B]
    M_Q_constraints = [0.5 * A - M * Q * n_p, M * Q * n_p - A]

    penalties.extend([max(0, -c) for c in N_P_constraints])
    penalties.extend([max(0, -c) for c in M_Q_constraints])

    return np.sum(penalties)

def combined_objective(x):
    return objective(x) + penalty(x)

# Define bounds for each variable
bounds = [(1, 50), (1, 50), (1, 50), (1, 50)]   # Decision variables: [M, N, n_t, n_p]

result_de = differential_evolution(combined_objective, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7)

# Ensure the final solution is integer
result_de.x = list(map(int, result_de.x))

print("Differential Evolution result:")
print(result_de)
