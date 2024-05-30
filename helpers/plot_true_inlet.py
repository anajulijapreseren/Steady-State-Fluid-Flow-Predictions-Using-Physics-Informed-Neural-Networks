import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to plot contours
def plot_contours(x, y, z, title, ax):
    if z.min() == z.max():
        contour_levels = np.linspace(z.min(), z.max() + 1, num=50)
    else:
        contour_levels = np.linspace(z.min(), z.max(), num=50)
    cp = ax.contourf(x, y, z, levels=contour_levels, cmap='viridis')
    plt.colorbar(cp, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Read the CSV file
data = pd.read_csv('combined_data.csv')

# Extract the data
x = data['x'].values
y = data['y'].values
u = data['u'].values
v = data['v'].values

# Create a grid
nx = len(np.unique(x))
ny = len(np.unique(y))
X = x.reshape((nx, ny))
Y = y.reshape((nx, ny))
U = u.reshape((nx, ny))
V = v.reshape((nx, ny))

# Plot the contours
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

plot_contours(X, Y, U, 'Velocity u', axs[0])
plot_contours(X, Y, V, 'Velocity v', axs[1])

plt.tight_layout()
plt.savefig('true_inlet.png')
plt.show()
