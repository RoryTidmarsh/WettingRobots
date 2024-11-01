import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
# import shapes

# parameters
L = 10.0 # size of box
rho = 4.2 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.23
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.15 # noise/randomness
max_num_neighbours= N

#Defining how much the particle turns when in the boundary
wall_distance = r0 +0.8 #detection radius
wall_turn = np.deg2rad(110) #The max angle to turn when wall_distance = 0
turn_factor = 10. #scaling for how close the particle turns to the wall. Higher = turns closer to the wall

#Defining parameters for a wall only in the x direction.
wall_x = 5.
wall_yMin = 1.
wall_yMax = 9.
from shapes import x_wall
# @numba.njit
x_wall1 = x_wall(wall_x, wall_yMax, wall_yMin, wall_distance)
filt = x_wall1.x_wall_filter

#Defining parameters for a rectangle
x_min, x_max, y_min, y_max = 4.,6.,4.,6.

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 


@numba.njit
def varying_angle_turn(dist, turn_factor):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        float: angle for the particle to turn
    """
    global wall_distance, wall_turn
    turn_angle = np.where(dist < wall_distance, wall_turn * np.exp(-turn_factor * dist), 0)
    return turn_angle

@numba.njit
def update(positions, angles, func):
    """Updates the positions and angles of the particles based on the function provided
    Args
        positions: 2d array of the positions of all the particles
        angles: 1d array of the angles of all the particles
        func: function that takes the position and angle of a particle and returns the new position and angle

    Returns:
        tuple of np.darrays: (new_positions, new_angles)
    """
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
        x_pos = positions[i,0]
        y_pos = positions[i,1]
         
        distance_to_wall = func(x_pos, y_pos) 
        # if there are neighbours, calculate average angle and noise/randomness       
        # if neighbour_angles:
        wall_turn = varying_angle_turn(dist=distance_to_wall,turn_factor=turn_factor)
        if count_neigh > 0:
            average_angle = np.mean(neigh_angles[:count_neigh])
            noise = eta * np.random.uniform(-np.pi, np.pi)
            
            
            if average_angle <= 0:
                new_angles[i] = average_angle - wall_turn + noise
            else:
                new_angles[i] = average_angle + wall_turn + noise
            
        else:
            # if no neighbours, keep current angle unless close to wall
            if average_angle <= 0:
                new_angles[i] = angles[i] - wall_turn + noise
            else:
                new_angles[i] = angles[i] + wall_turn + noise
            # Make the particle turn around based on the wall_turn parameter
            
        
        # update position based on new angle
        # new position from speed and direction   
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        # boundary conditions of bo
        new_positions[i] %= L

    return new_positions, new_angles

def animate(frames):
    print(frames)
    
    global positions, angles
    # empty arrays to hold updated positions and angles
    
    new_positions, new_angles = update(positions, angles, filt)
        
    # update global variables
    positions = new_positions
    angles = new_angles
    
    # plotting
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(angles), np.sin(angles), angles)

    
    return qv,

fig, ax = plt.subplots(figsize = (6, 6))  
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ax = x_wall1.plot_x_wall_boundary(ax=ax)
ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
ax.legend(loc = "upper right")
# ani.save(f'code/wall implemetation/figures/Vicek_varying_wall_turn_p={rho:.2f}.gif', writer='imagemagick', fps=30)
plt.show()









