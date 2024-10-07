import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('k_results_various_A_B_new.xlsx')

x = data['k']
y_1 = data['n_r']
y_2 = data['M']
y_3 = data['N']
y_4 = data['value']
y_5 = data['capacity_check']
# A = data['length_check']
# B = data['width_check']
A = data['A']
B = data['B']


plt.figure(figsize=(8, 5))


plt.plot(x, y_4, c='r', label='Total travel distance', marker='^')
plt.xlabel('Demand (k)')
plt.ylabel('Total Travel Distance (d)')
plt.title('Total average distance performance: the finite case')

# plt.plot(x, y_1*20, c='orange', label='Block length (BL_l)', marker='o')
# plt.xlabel('Demand (k)')
# plt.ylabel('Block length (ft)')
# plt.title('Demand (k) vs. Block length(BL_l)')

# plt.plot(x, y_3, c='b', label='Block rows (N)', marker='^')
# plt.plot(x, y_2, c='olive', label='Block columns (M)', alpha=0.5, marker='o')
# plt.plot(x, y_2 * y_3, c='maroon', label='Block numbers (MN)', marker='*')
# plt.xlabel('Demand (k)')
# plt.ylabel('Numbers of blocks')
# plt.title('Demand (k) vs. Block numbers(M & N)')

# plt.plot(x, A, c='b', label='Yard length(A)', marker='o')
# plt.plot(x, B, c='r', label='Yard width (B)', marker='o')
# plt.axhline(y=1928, color='b', linestyle='--', label='True length')
# plt.axhline(y=765, color='r', linestyle='--', label='True width')
# plt.xlabel('Demand (k)')
# plt.ylabel('Yard shape (ft)')
# plt.title('Demand (k) vs. Yard shape (A or B)')

# plt.plot(x, A*B, c='purple', label='Yard area (A*B)', marker='o')
# plt.axhline(y=1470920, color='purple', linestyle='--', label='True landscape')
# plt.xlabel('Demand (k)')
# plt.ylabel('Yard area (ft^2)')
# plt.title('Demand (k) vs. Yard area (A*B)')

plt.legend()
plt.show()