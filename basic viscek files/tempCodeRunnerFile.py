import numpy as np
import matplotlib.pyplot as plt

# Define the positions of the two charges as complex numbers
charge1_pos = -2 + 0j  # Position of the first charge (e.g., -2 on the x-axis)
charge2_pos = 2 + 0j   # Position of the second charge (e.g., 2 on the x-axis)

# Define the strength of each charge
q1 = 1
q2 = 1

# Create a grid of points in the complex plane
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y  # Complex plane representation

# Calculate the field due to each charge
# Field from charge 1
epsilon = 1e-3  # Small value to avoid division by zero
F1 = q1 * (Z - charge1_pos) / (np.abs(Z - charge1_pos)**3 + epsilon)

# Field from charge 2
F2 = q2 * (Z - charge2_pos) / (np.abs(Z - charge2_pos)**3 + epsilon)

# Total field as the vector sum of the fields from both charges
F_total = F1 + F2

# Separate the field into real and imaginary parts for plotting
Fx, Fy = np.real(F_total), np.imag(F_total)

# Plot the force field using streamlines for a smoother appearance
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Fx, Fy, color=np.sqrt(Fx**2 + Fy**2), cmap='inferno', linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Symmetric Field of Two Positive Charges")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.colorbar(label="Field strength")
plt.grid()
plt.show()
