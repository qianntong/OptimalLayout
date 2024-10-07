import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_excel('k_results_fixed_A_B.xlsx')
# data = pd.read_excel('k_results_various_A_B.xlsx')
# data = pd.read_excel('k_results_various_A_B_large_sample.xlsx')


x = data['k']
y_1 = data['n_r']
y_2 = data['M']
y_3 = data['N']
z = data['value']


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y_1, z, c='r', marker='o',label='columns within each block unit')
ax.scatter(x, y_2, z, c='g', marker='^',label='numbers of parking block columns')
ax.scatter(x, y_3, z, c='b', marker='s',label='numbers of parking block rows')

ax.set_xlabel("Expected Demand")
ax.set_ylabel("Block Columns")
ax.set_zlabel("Total Distances")
ax.set_title("Decision Making Process for Optimal Layout")
ax.legend()
plt.show()
