import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
# parameters
L = 10.0 # size of box
rho = 3.5 # density
N = int(rho * L**2) # number of particles
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 0.5
v0 = r0 / deltat * factor # velocity
iterations = 100 # animation frames
eta = 0.05 # noise/randomness
max_num_neighbours = N # it could be less...

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) # from 0 to 2pi rad

@numba.njit
def update(positions, angles):
    # empty arrays to hold updated positions and angles
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    neigh_angles = np.empty(max_num_neighbours)
    # loop over all particles
    for i in range(N):
        # list of angles of neighbouring particles
        # neighbour_angles = []
        count_neigh = 0

        # distance to other particles
        for j in range(N):
            distance = np.linalg.norm(positions[i] - positions[j])
            # if within interaction radius add angle to list
            if distance < r0:
                # neighbour_angles[].append(angles[j])
                neigh_angles[count_neigh] = angles[j]
                count_neigh += 1
         
        # if there are neighbours, calculate average angle and noise/randomness       
        # if neighbour_angles:
        if count_neigh > 0:
            average_angle = np.mean(neigh_angles[:count_neigh])
            noise = eta * np.random.uniform(-np.pi, np.pi)
            new_angles[i] = average_angle + noise # updated angle with noise
        else:
            # if no neighbours, keep current angle
            new_angles[i] = angles[i]
        
        # update position based on new angle
        # new position from speed and direction   
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        # boundary conditions of bo
        new_positions[i] %= L

    return new_positions, new_angles


def animate(frames):
    print(frames)
    
    global positions, angles
    # we were here...
    new_positions, new_angles = update(positions,angles)
        
    # update global variables
    positions = new_positions
    angles = new_angles
    
    # plotting
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(angles), np.sin(angles), angles)
    return qv,
 
fig, ax = plt.subplots(figsize = (6, 6))   
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
plt.show()