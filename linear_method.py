import numpy as np
from scipy.optimize import minimize

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

# Estimation for decision variable initial values
x0 = np.array([1, 1, 2, 2]) # [M, N, n_t, n_p]

def d_t(n_t):
    return 2 * (k * l + s - B) + 2 * n_t * P

def d_y(M, N, n_p):
    return 2 * (0.5 * n_p * P + 0.5 * BL_l) + 2 * (1.5 * n_p * P + BL_w)

def d_g(M, N, n_p):
    return 2 * n_p * P

def objective(x):
    M, N, n_t, n_p = x
    return (d_t(n_t) / v_hostler +
            d_y(M, N, n_p) / v_hostler +
            d_g(M, N, n_p) / v_truck)

def constraint_d_t(x, b):
    _, _, n_t, _ = x
    return b - d_t(n_t) / v_hostler

def constraint_d_y(x, c):
    M, N, _, n_p = x
    return c - d_y(M, N, n_p) / v_hostler

def constraint_d_g(x, d):
    M, N, _, n_p = x
    return d - d_g(M, N, n_p) / v_truck

def constraint_n_tPQ(x, f):
    _, _, n_t, _ = x
    return f - n_t * P * Q

def constraint_c_p(x):
    M, N, _, _ = x
    return (c_p * M * N * BL_w * BL_l / S_p) - np.sum(delta_k * n_k)

def constraint_N_P(x):
    _, N, _, n_p = x
    return [0.5 * B - N * P * n_p, N * P * n_p - B]

def constraint_M_Q(x):
    M, _, _, n_p = x
    return [0.5 * A - M * Q * n_p, M * Q * n_p - A]

def constraint_non_negative(x):
    return x

# Non-linear constraints
constraints = [
    {"type": "ineq", "fun": constraint_d_t, "args": (b,)},
    {"type": "ineq", "fun": constraint_d_y, "args": (c,)},
    {"type": "ineq", "fun": constraint_d_g, "args": (d,)},
    {"type": "ineq", "fun": constraint_n_tPQ, "args": (f,)},
    {"type": "ineq", "fun": constraint_c_p},
    {"type": "ineq", "fun": constraint_N_P},
    {"type": "ineq", "fun": constraint_M_Q},
    {"type": "ineq", "fun": constraint_non_negative},
]

# Optimizing using the function of minimize with SciPy for different methods and increased iterations
result_slsqp = minimize(
    objective,
    x0,
    constraints=constraints,
    method='SLSQP',
    options={'maxiter': 1000}
)

result_cobyla = minimize(
    objective,
    x0,
    constraints=constraints,
    method='COBYLA',
    options={'maxiter': 1000}
)

result_lbfgsb = minimize(
    objective,
    x0,
    constraints=constraints,
    method='L-BFGS-B',
    options={'maxiter': 1000}
)

result_trust_constr = minimize(
    objective,
    x0,
    constraints=constraints,
    method='trust-constr',
    options={'maxiter': 1000}
)

print("SLSQP result:")
print(result_slsqp)

print("COBYLA result:")
print(result_cobyla)

print("L-BFGS-B result:")
print(result_lbfgsb)

print("trust-constr result:")
print(result_trust_constr)