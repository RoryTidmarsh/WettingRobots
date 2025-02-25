import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
from pycircular.stats import periodic_mean_std
import os

# parameters
L = 50 # size of box
rho = 1 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.1  # noise/randomness
max_num_neighbours= 100


# Thick wall parameters
y_min = 0
y_max = L
width = r0
x_min, x_max = L/2 - width/2, L/2 + width/2  # Thickness of r0

wall_distance = r0  # interaction distance from wall
turn_factor = 0.2
step_num = 0

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))    #or just use pycircular module

#### Wall funcitons ####
@numba.njit
def dist_to_rect(x_pos, y_pos, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
    """
    Finds the distance of the particle to the nearest wall of a rectangle
    and the unit vector pointing toward that wall.
    
    Args:
        x_pos (float): x position of the particle
        y_pos (float): y position of the particle
        x_min, x_max, y_min, y_max: Rectangle boundaries (optional)
    
    Returns:
        tuple: (distance, unit_vector) where:
            - distance (float): distance to nearest wall, or -1 if inside rectangle
            - unit_vector (ndarray): 2D unit vector pointing toward the nearest wall
    """
    # Check if point is inside rectangle
    if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
        return -1.0, 0+0j
    
    # Initialize variables with numeric types
    dx = 0.0
    dy = 0.0
    dir_x = 0.0
    dir_y = 0.0
    
    # Calculate horizontal distance and component of direction vector
    if x_pos < x_min:
        dx = x_min - x_pos
        dir_x = 1.0  # Direction to the right
    elif x_pos > x_max:
        dx = x_pos - x_max
        dir_x = -1.0  # Direction to the left
    
    # Calculate vertical distance and component of direction vector
    if y_pos < y_min:
        dy = y_min - y_pos
        dir_y = 1.0  # Direction is upward
    elif y_pos > y_max:
        dy = y_pos - y_max
        dir_y = -1.0  # Direction is downward
    
    # Calculate Euclidean distance
    distance = np.sqrt(dx**2 + dy**2)
    
    # Create direction vector with explicit type
    direction = dir_x*dx + 1j* dir_y*dy
    
    # Normalize for unit vector
    if distance > 0:
        unit_vector = direction / distance
    else:
        unit_vector = 0 +0j
    
    return distance, -1*unit_vector

def adjust_initial_positions(positions = positions, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
    """Adjust initial positions of particles to be within the rectangle boundaries."""
    for i in range(len(positions)):
        while dist_to_rect(positions[i][0], positions[i][1], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)[0] <0:
            positions[i] = np.random.uniform(0, L, size=(1, 2))
    return positions
positions = adjust_initial_positions(positions)

@numba.njit
def rect_angle_turn(x_pos, y_pos, turn_factor, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        x_pos, y_pos (float): particle positions
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        complex_turn: complex number representing the angle
        inside: flag indicating if the particle is inside the rectangle
    """
    dist, r_hat = dist_to_rect(x_pos, y_pos, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    if 0 <= dist < wall_distance:
        # Ensure r_hat is scaled by turn_factor/dist
        complex_turn = (turn_factor/dist) * r_hat
    
    else:
        complex_turn = 0
        
    return complex_turn

def plot_rect(ax):
    rectangle = plt.Rectangle((x_min, y_min),
                              x_max - x_min,
                              y_max - y_min, 
                              edgecolor='black',
                              facecolor='lightgray',
                              alpha=0.3)
    ax.add_patch(rectangle)
    return ax

#### Cell Searching####
# cell list
cell_size = 1.*r0
lateral_num_cells = int(L/cell_size)
total_num_cells = lateral_num_cells**2
max_particles_per_cell = int(rho*cell_size**2*10)

@numba.njit
def get_cell_index(pos, cell_size, num_cells):
    return int(pos[0] // cell_size) % num_cells, int(pos[1] // cell_size) % num_cells
 
@numba.njit(parallel=True)
def initialise_cells(positions, cell_size, num_cells, max_particles_per_cell):
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


@numba.njit(parallel=True)
def update(positions, angles, cell_size, num_cells, max_particles_per_cell):
    
    N = positions.shape[0]
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    
    
    # Initialize cell lists
    cells, cell_counts = initialise_cells(positions, cell_size, num_cells, max_particles_per_cell)

    for i in numba.prange(N):  # Parallelize outer loop
        neigh_angles = np.empty(max_num_neighbours)
        count_neigh = 0
        # neigh_angles.fill(0)  # Reset neighbor angles

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

                        # Periodic Interaction
                        dx = dx - L * np.round(dx/L)
                        dy = dy - L * np.round(dy/L)
                    
                        distance_sq = dx * dx + dy * dy
                        if distance_sq < r0 * r0:  # Compare with squared radius
                            if count_neigh < max_num_neighbours:
                                neigh_angles[count_neigh] = angles[j]
                                count_neigh += 1

        x_pos = positions[i,0]
        y_pos = positions[i,1]
          
        # Calculate the angle to turn due to the wall
        # wall_turn = varying_angle_turn(x_pos, y_pos,turn_factor=turn_factor, wall_yMin=wall_yMin, wall_yMax=wall_yMax)
        if (x_min-x_max ==0)&(y_min-y_max ==0):   # no wall case
            wall_turn = 0
        else:
            wall_turn = rect_angle_turn(x_pos,y_pos, turn_factor, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        noise = eta * np.random.uniform(-np.pi, np.pi)
        # if there are neighbours, calculate average angle      
        if count_neigh > 0:
            average_angle = np.mean(np.exp(neigh_angles[:count_neigh]*1.0j)) #angle expressed as a complex number
            new_complex = average_angle + wall_turn #wall_turn = 0 if not in boundary
            
        else:
            new_complex = np.exp(angles[i]*1j) + wall_turn
            
        new_angles[i] = noise + np.angle(new_complex)
        
        # new position from speed and direction   
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        dist, _ = dist_to_rect(new_positions[i][0], new_positions[i][1], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        if dist <0:
            new_positions[i] = positions[i] # if particle is inside the rectangle, keep the old
        # boundary conditions of box
        new_positions[i] %= L

    return new_positions, new_angles

def animate(frames):
    # print(frames)
    global positions, angles, step_num
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell)


    # Update global variables
    positions = new_positions.copy()
    angles = new_angles.copy()
    step_num +=1
    print(step_num)
    
    #Update the quiver plot  
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    return qv,
 
## Showing the animation
figwidth = 7
totalheight = 6
fig, ax = plt.subplots(figsize = (figwidth, totalheight))   
# arc_angle = np.pi/6
# start_angle = -arc_angle/2

ax.set_title(f"Vicsek Model in Python. $\\rho = {rho}$, $\\eta = {eta}$")
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# Add a color bar
cbar = fig.colorbar(qv, ax=ax, label="Angle (radians)")
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

ax = plot_rect(ax)
# ani = FuncAnimation(fig, animate, frames = range(1, int(iterations/10)), fargs=arc_angle, interval = 5, blit = True)
ani = FuncAnimation(fig, animate, frames=range(1, int(iterations/10)), fargs=(), interval=5, blit=True)
ax.legend(loc = "upper right")
# ani.save(f'figures/Vicsek_={rho}_eta={eta}.gif', writer='pillow', fps=30)
plt.show()
# np.savez_compressed(f'{os.path.dirname(__file__)}/wall_size_experiment/finalstate.npz', Positions = positions, Orientation = angles)

