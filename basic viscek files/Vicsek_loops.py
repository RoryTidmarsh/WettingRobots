import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
from numba import float64
# parameters
L = 60.0 # size of box
rho = 0.5 # density
N = int(rho * L**2) # number of particles
print("the number  of particles is ", N)
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 0.2
v0 = r0 / deltat * factor # velocity
iterations = 100 # animation frames
eta = 0.5 # noise/randomness
max_num_neighbours = N # it could be less...

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) # from 0 to 2pi rad

av_frames_angles = 10
num_frames_av_angles = np.empty(av_frames_angles)
num_frames_av_angles2 = np.empty(av_frames_angles)
t = 0

# cell list
cell_size = 2*r0
lateral_num_cells = int(L/cell_size)
total_num_cells = lateral_num_cells**2
max_particles_per_cell = int(rho*cell_size**2*10)

@numba.njit
def get_cell_index(pos, cell_size, num_cells):
    return int(pos[0] // cell_size) % num_cells, int(pos[1] // cell_size) % num_cells
# 
@numba.njit(parallel=True)
def initialize_cells(positions, cell_size, num_cells, max_particles_per_cell):
    # Create cell arrays
    cells = np.full((num_cells, num_cells, max_particles_per_cell), -1, dtype=np.int32)  # -1 means empty
    cell_counts = np.zeros((num_cells, num_cells), dtype=np.int32)
    
    # Populate cells with particle indices
    for i in  numba.prange(positions.shape[0]):
        cell_x, cell_y = get_cell_index(positions[i], cell_size, num_cells)
        idx = cell_counts[cell_x, cell_y]
        if idx < max_particles_per_cell:
            cells[cell_x, cell_y, idx] = i  # Add particle index to cell
            cell_counts[cell_x, cell_y] += 1  # Update particle count in this cell
    return cells, cell_counts

@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))
average_angles = []
average_angles2 = []

@numba.njit(parallel=True)
def update(positions, angles, cell_size, num_cells, max_particles_per_cell):
    N = positions.shape[0]
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    neigh_angles = np.empty(max_num_neighbours)
    
    # Initialize cell lists
    cells, cell_counts = initialize_cells(positions, cell_size, num_cells, max_particles_per_cell)

    for i in numba.prange(N):  # Parallelize outer loop
        count_neigh = 0
        neigh_angles.fill(0)  # Reset neighbor angles

        # Get particle's cell, ensuring indices are integers
        cell_x, cell_y = get_cell_index(positions[i], cell_size, num_cells)

        # Check neighboring cells (3x3 neighborhood)
        for cell_dx in (-1, 0, 1):
            for cell_dy in (-1, 0, 1):
                # Ensure neighbor_x and neighbor_y are integers
                neighbor_x = int((cell_x + cell_dx) % num_cells)
                neighbor_y = int((cell_y + cell_dy) % num_cells)

                # Check each particle in the neighboring cell
                for idx in range(cell_counts[neighbor_x, neighbor_y]):
                    j = cells[neighbor_x, neighbor_y, idx]
                    if i != j:  # Avoid self-comparison
                        # Calculate squared distance for efficiency
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance_sq = dx * dx + dy * dy
                        if distance_sq < r0 * r0:  # Compare with squared radius
                            if count_neigh < max_num_neighbours:
                                neigh_angles[count_neigh] = angles[j]
                                count_neigh += 1

        # Apply noise using Numba-compatible randomness
        noise = eta * (np.random.random() * 2 * np.pi - np.pi)

        # Calculate new angle
        if count_neigh > 0:
            average_angle = np.angle(np.sum(np.exp(1j * neigh_angles[:count_neigh])))
            new_angles[i] = average_angle + noise
        else:
            new_angles[i] = angles[i] + noise

        # Update position based on new angle
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        new_positions[i] %= L  # Apply boundary conditions

    return new_positions, new_angles

def animate(frames):
    print(frames)
    global positions, angles, t, num_frames_av_angles
    
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell)
    
    # Store the new angles in the num_frames_av_angles array
    average_angles2.append(average_angle(new_angles))
    num_frames_av_angles[t] = average_angle(new_angles)
    if t == av_frames_angles - 1:  # Check if we've filled the array
        average_angles.append(average_angle(num_frames_av_angles))
        t = 0  # Reset t
        num_frames_av_angles = np.empty(av_frames_angles)  # Reinitialize the array
    else:
        t += 1  # Increment t
        
    # Update global variables
    positions = new_positions
    angles = new_angles
    
    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    return qv,


fig, ax = plt.subplots(figsize = (6, 6))   
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ax.grid()
# for i in range(10000):
#     animate(i)
ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
plt.show()


# fig, ax2 = plt.subplots()
# times = np.arange(0,len(average_angles))*av_frames_angles
# ax2.plot(times, average_angles, label = "10 frames.")
# ax2.plot(np.arange(len(average_angles2)), average_angles2, label = "1 frame")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Angle (radians)")
# ax2.set_title("Alignment, averaging from different number of frames.")
# ax2.legend()
# plt.show()
