import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
L = 10.0 # size of box
rho = 1.0 # density
N = int(rho * L**2) # number of particles
r0 = 0.75 # interaction radius
deltat = 1.0 # time step
factor = 0.3
v0 = r0 / deltat * factor # velocity
iterations = 200 # animation frames
eta = 0.05 # noise/randomness

#Defining parameters for a wall only in the x direction.
wall_x = 5.
wall_yMin = 2.
wall_yMax = 8.
wall_distance = 0.75
wall_turn = np.deg2rad(30)


# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

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

def plot_x_wall_boundary(ax):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
    # Boundary left and right of the wall within vertical bounds
    ax.plot([wall_x - wall_distance, wall_x - wall_distance], [wall_yMin, wall_yMax], 'b--', lw=2, label=f'Boundary at {wall_distance} units')
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

def animate(frames):
    print(frames)
    
    global positions, angles
    # empty arrays to hold updated positions and angles
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    
    # loop over all particles
    for i in range(N):
        # list of angles of neighbouring particles
        neighbour_angles = []
        # distance to other particles
        for j in range(N):
            distance = np.linalg.norm(positions[i] - positions[j])
            # if within interaction radius add angle to list
            if distance < r0:
                neighbour_angles.append(angles[j])

        #Checking if the particles are within distance_away from the wall
        x_pos = positions[i,0]
        y_pos = positions[i,1]
         
        distance_to_wall = x_wall_filter(x_pos, y_pos)
        # if there are neighbours, calculate average angle and noise/randomness      
       

        if neighbour_angles:
            average_angle = np.mean(neighbour_angles)
            noise = eta * np.random.uniform(-np.pi, np.pi)
            if distance_to_wall < wall_distance:
                if average_angle <= 0:
                    new_angles[i] = average_angle - wall_turn + noise
                else:
                    new_angles[i] = average_angle + wall_turn + noise
            # Make the particle turn around based on the wall_turn parameter
                 # updated angle with noise
            else:
                new_angles[i] = average_angle + noise
        else:
            # if no neighbours, keep current angle

            if distance_to_wall < wall_distance:
                if average_angle <= 0:
                    new_angles[i] = average_angle - wall_turn + noise
                else:
                    new_angles[i] = average_angle + wall_turn + noise
                # Make the particle turn around based on the wall_turn parameter
            else:
                new_angles[i] = angles[i]
        # update position based on new angle
        # new position from speed and direction   
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        # boundary conditions of bo
        new_positions[i] %= L

        
        
    # update global variables
    positions = new_positions
    angles = new_angles
    
    # plotting
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(angles), np.sin(angles), angles)

    
    return qv,
 
fig, ax = plt.subplots(figsize = (6, 6))   

ax = plot_x_wall_boundary(ax)
ax.set_title(f"{N} particles, turning near a wall. Loop method.")
ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall")
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")

ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
ax.legend()
# ani.save('code/wall implemetation/Vicek_looped_walls.gif', writer='imagemagick', fps=30)
plt.show()