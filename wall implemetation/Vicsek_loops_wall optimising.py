import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
# parameters
L = 10.0 # size of box
rho = 3 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 1
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.15 # noise/randomness
max_num_neighbours= N


#Defining parameters for a wall only in the x direction.
wall_x = 5.
wall_yMin = 1.
wall_yMax = 9.
wall_distance = r0 +0.8
wall_turn = np.deg2rad(110)
turn_factor = 10.

#Defining parameters for a rectangle
x_min, x_max, y_min, y_max = 4.,6.,4.,6.

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

@numba.njit
def x_wall_filter(x_pos,y_pos):
    """Finds the distance of the arrow to the wall.

    Args:
        x_pos (float): x_position of 1 particle
        y_pos (flaot): y_position of 1 particle
    Returns:
        distance_to_wall: distance of the particle to the wall
    """
    if y_pos > wall_yMax:
        #particle above the wall
        distance_to_wall = np.sqrt((x_pos-wall_x)**2 + (y_pos-wall_yMax)**2)
    elif y_pos < wall_yMin:
        #particle below the wall
        distance_to_wall = np.sqrt((x_pos-wall_x)**2 + (y_pos-wall_yMin)**2)
    else:
        #particle level with the wall
        distance_to_wall = np.abs(x_pos-wall_x)
    return distance_to_wall

@numba.njit
def varying_angle_turn(dist, turn_factor):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        turn_angle: angle for the particle to turn
    """
    global wall_distance, wall_turn
    turn_angle = np.where(dist < wall_distance, wall_turn * np.exp(-turn_factor * dist), 0)
    return turn_angle

def plot_x_wall_boundary(ax):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
    # Boundary left and right of the wall within vertical bounds
    ax.plot([wall_x - wall_distance, wall_x - wall_distance], [wall_yMin, wall_yMax], 'b--', lw=2, label=f'Boundary at {wall_distance:.2f}')
    ax.plot([wall_x + wall_distance, wall_x + wall_distance], [wall_yMin, wall_yMax], 'b--', lw=2)
    
    # Boundary above the wall (top circle segment)
    theta = np.linspace(0, np.pi, 100)  # For the top part of the wall
    top_circle_x = wall_x + wall_distance * np.cos(theta)
    top_circle_y = wall_yMax + wall_distance * np.sin(theta)
    ax.plot(top_circle_x, top_circle_y, 'b--', lw=2)
    
    # Boundary below the wall (bottom circle segment)
    theta = np.linspace(np.pi, 2 * np.pi, 100)  # For the bottom part of the wall
    bottom_circle_x = wall_x + wall_distance * np.cos(theta)
    bottom_circle_y = wall_yMin + wall_distance * np.sin(theta)
    ax.plot(bottom_circle_x, bottom_circle_y, 'b--', lw=2)

    #plot the wall
    ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall")
    return ax

def position_filter(positions, filter_func):
    """Adjusts the initial positions so they are no longer within the boundary.

    Args:
        positions (numpy array): array of positions
        filter_func (function): function that calculates the distance to the wall

    Returns:
        positions
    """
    for i in range(len(positions)):
        # Check if the particle is too close to the wall
        while filter_func(positions[i][0], positions[i][1]) <= wall_distance:
            # Regenerate position until it is far enough from the wall
            positions[i] = np.random.uniform(0, L, size=(1, 2))
    return positions

positions = position_filter(positions, x_wall_filter)

@numba.njit
def update(positions, angles, func):
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
    
    new_positions, new_angles = update(positions, angles, x_wall_filter)
        
    # update global variables
    positions = new_positions
    angles = new_angles
    
    # plotting
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(angles), np.sin(angles), angles)

    
    return qv,
 
fig, ax = plt.subplots(figsize = (6, 6))   

ax = plot_x_wall_boundary(ax)
ax.set_title(f"{N} particles, turning near a wall. Varying angle with wall distance.")

qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")

ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
ax.legend(loc = "upper right")
# ani.save(f'code/wall implemetation/figures/Vicek_varying_wall_turn_p={rho:.2f}.gif', writer='imagemagick', fps=30)
plt.show()