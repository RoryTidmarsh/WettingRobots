"""animation of Vicsek model with walls. Currently in a reduced system to increase the run speed

There are no outputs in this file
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
import os
import tqdm
#Save the end state as a .png
save_fig = False

# parameters
L = 64 # size of box
rho = 1.2 # density
N = int(rho * L**2) # number of particles
r0 = 1 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 1000 # animation frames
eta = 0.15  # noise/randomness
max_num_neighbours= 100


# Defining parameters for walls in both x and y directions.
wall_x1 = 1
wall_x2 = L - 1
wall_y1 = 1
wall_y2 = L - 1
wall_distance = r0
turn_factor = 0.2
step_num = 0

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 


# Ensure all particles are spawned within the barriers
confined = True
if confined:
    positions = np.random.uniform([wall_x1, wall_y1], [wall_x2, wall_y2], size=(N, 2))
@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))    #or just use pycircular module

#### Wall funcitons ####
@numba.njit
def x_wall_filter(x_pos,y_pos, wall_yMax, wall_yMin, wall_x = wall_x1):
    """Finds the distance of the arrow to the wall.

    Args:
        x_pos (float): x_position of 1 particle
        y_pos (float): y_position of 1 particle
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
def y_wall_filter(x_pos, y_pos, wall_xMax, wall_xMin, wall_y=wall_y1):
    """Finds the distance of the particle to the wall in the y direction.""" 
    if x_pos > wall_xMax:
        distance_to_wall = np.sqrt((x_pos - wall_xMax)**2 + (y_pos - wall_y)**2)
    elif x_pos < wall_xMin:
        distance_to_wall = np.sqrt((x_pos - wall_xMin)**2 + (y_pos - wall_y)**2)
    else:
        distance_to_wall = np.abs(y_pos - wall_y)
    return distance_to_wall

@numba.njit
def varying_angle_turn(x_pos, y_pos, turn_factor, wall_y1, wall_y2, wall_x):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary."""
    x_dist = x_pos - wall_x
    dist = x_wall_filter(x_pos, y_pos, wall_y2, wall_y1, wall_x)
    if dist < wall_distance:
        if wall_y1 <= y_pos <= wall_y2:
            y_dist = 0
        elif wall_y1 > y_pos:
            y_dist = y_pos - wall_y1
        elif wall_y2 < y_pos:
            y_dist = y_pos - wall_y2

        r_hat = (x_dist + y_dist * 1j) / (x_dist**2 + y_dist**2)
        complex_turn = (turn_factor / dist) * r_hat
    else:
        complex_turn = 0
    return complex_turn

@numba.njit
def varying_angle_turn_y(x_pos, y_pos, turn_factor, wall_x1, wall_x2, wall_y):
    """Tells the particle how much to turn when it is within the boundary in the y direction."""
    y_dist = y_pos - wall_y
    dist = y_wall_filter(x_pos, y_pos, wall_x2, wall_x1, wall_y)
    if dist < wall_distance:
        if wall_x1 <= x_pos <= wall_x2:
            x_dist = 0
        elif wall_x1 > x_pos:
            x_dist = x_pos - wall_x1
        elif wall_x2 < x_pos:
            x_dist = x_pos - wall_x2

        r_hat = (x_dist + y_dist * 1j) / (x_dist**2 + y_dist**2)
        complex_turn = (turn_factor / dist) * r_hat
    else:
        complex_turn = 0
    return complex_turn

def plot_x_wall(ax, wall_x, wall_y1, wall_y2, wall_color="blue", boundary=True, walpha=1):
    """Plots the boundary based on the initial dimensions of the wall set in the x direction."""
    if boundary:
        ax.plot([wall_x - wall_distance, wall_x - wall_distance], [wall_y1, wall_y2], 'b--', lw=2, label=f'Boundary at {wall_distance:.2f}')
        ax.plot([wall_x + wall_distance, wall_x + wall_distance], [wall_y1, wall_y2], 'b--', lw=2)
        theta = np.linspace(0, np.pi, 100)
        top_circle_x = wall_x + wall_distance * np.cos(theta)
        top_circle_y = wall_y2 + wall_distance * np.sin(theta)
        ax.plot(top_circle_x, top_circle_y, 'b--', lw=2)
        theta = np.linspace(np.pi, 2 * np.pi, 100)
        bottom_circle_x = wall_x + wall_distance * np.cos(theta)
        bottom_circle_y = wall_y1 + wall_distance * np.sin(theta)
        ax.plot(bottom_circle_x, bottom_circle_y, 'b--', lw=2)
    ax.plot([wall_x, wall_x], [wall_y1, wall_y2], label="wall", color=wall_color, alpha=walpha)
    return ax

def plot_y_wall(ax, wall_y, wall_x1, wall_x2, wall_color="red", boundary=True, walpha=1):
    """Plots the boundary based on the initial dimensions of the wall set in the y direction."""
    if boundary:
        ax.plot([wall_x1, wall_x2], [wall_y - wall_distance, wall_y - wall_distance], 'r--', lw=2, label=f'Boundary at {wall_distance:.2f}')
        ax.plot([wall_x1, wall_x2], [wall_y + wall_distance, wall_y + wall_distance], 'r--', lw=2)
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
        left_circle_x = wall_x1 + wall_distance * np.cos(theta)
        left_circle_y = wall_y + wall_distance * np.sin(theta)
        ax.plot(left_circle_x, left_circle_y, 'r--', lw=2)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
        right_circle_x = wall_x2 + wall_distance * np.cos(theta)
        right_circle_y = wall_y + wall_distance * np.sin(theta)
        ax.plot(right_circle_x, right_circle_y, 'r--', lw=2)
    ax.plot([wall_x1, wall_x2], [wall_y, wall_y], label="wall", color=wall_color, alpha=walpha)
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
def update(positions, angles, cell_size, num_cells, max_particles_per_cell, wall_y1, wall_y2, wall_x1, wall_x2):
    
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
          
        # Calculate the angle to turn due to the walls
        wall_turn1 = varying_angle_turn(x_pos, y_pos, turn_factor, wall_y1, wall_y2, wall_x1)
        wall_turn2 = varying_angle_turn(x_pos, y_pos, turn_factor, wall_y1, wall_y2, wall_x2)
        wall_turn3 = varying_angle_turn_y(x_pos, y_pos, turn_factor, wall_x1, wall_x2, wall_y1)
        wall_turn4 = varying_angle_turn_y(x_pos, y_pos, turn_factor, wall_x1, wall_x2, wall_y2)
        wall_turn = wall_turn1 + wall_turn2 + wall_turn3 + wall_turn4

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
        # boundary conditions of box
        new_positions[i] %= L

    return new_positions, new_angles

def animate(frames, wall_y1, wall_y2, wall_x1, wall_x2):
    # print(frames)
    global positions, angles, step_num
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell, wall_y1, wall_y2, wall_x1, wall_x2)


    # Update global variables
    positions = new_positions.copy()
    angles = new_angles.copy()
    step_num +=1
    # print(step_num)
    
    #Update the quiver plot  
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    return qv,
 
## Showing the animation
text_width = 3.25  # inches (single column width)
fig_width = text_width
fig_height = 0.8 * fig_width  # for standard plots
width_2_subplot = fig_width/2 + 0.25  # for side-by-side subplots
height_2_subplot = 0.75 * width_2_subplot
height_cbar_2_subplot = 0.75 * width_2_subplot
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.constrained_layout.use': True,
    'figure.autolayout': False,
    'axes.xmargin': 0.02,
    'axes.ymargin': 0.02,
    'figure.subplot.left': 0.12,
    'figure.subplot.right': 0.97,
    'figure.subplot.bottom': 0.12,
    'figure.subplot.top': 0.92,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})
fig, ax = plt.subplots(figsize = (fig_width, fig_width))   
ax = plot_x_wall(ax, wall_x1, wall_y1, wall_y2, boundary=False)
ax = plot_x_wall(ax, wall_x2, wall_y1, wall_y2, boundary=False)
ax = plot_y_wall(ax, wall_y1, wall_x1, wall_x2, boundary=False)
ax = plot_y_wall(ax, wall_y2, wall_x1, wall_x2, boundary=False)
## Double wall at the endgs of the system
# ax.axvline(x=x1,ymax=wall_yMax, ymin=wall_yMin, color = "black", linestyle = "--", lw = 2)
# ax.axvline(x=x2,ymax=wall_yMax, ymin=wall_yMin, color = "black", linestyle = "--", lw = 2)

# ax.set_title(f"Vicsek Model in Python. $\\rho = {rho}$, $\\eta = {eta}$")
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
## Add a color bar
# cbar = fig.colorbar(qv, ax=ax, label="Angle (radians)")
# cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
# cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

for i in tqdm.tqdm(range(0, 1500)):
    animate(i, wall_y1, wall_y2, wall_x1, wall_x2)

ani = FuncAnimation(fig, animate, frames= iterations, interval = 5, blit = True, fargs = (wall_y1,wall_y2, wall_x1, wall_x2))
if wall_y2 - wall_y1 == L:
    ax.legend(loc = "upper right")
ax.set_aspect("equal")
ax.set_xlabel(r"x ($R_0$)")
ax.set_ylabel(r"y ($R_0$)")
ax.set_yticks(np.linspace(0,L,5))
ax.set_xticks(np.linspace(0,L,5))
ax.set_ylim(0,L)
ax.set_xlim(0,L)
# ani.save(f'figures/Vicsek_={rho}_eta={eta}.gif', writer='pillow', fps=30)
plt.show()
# np.savez_compressed(f'{os.path.dirname(__file__)}/wall_size_experiment/finalstate.npz', Positions = positions, Orientation = angles)

if save_fig:
    fig, ax = plt.subplots(figsize = (fig_width*2/3, fig_width*2/3)) 
    if wall_y1 != wall_y2:
        ax = plot_x_wall(ax, wall_x1, wall_y1, wall_y2, boundary=False,wall_color="r")
        ax = plot_y_wall(ax, wall_y1, wall_x1, wall_x2, boundary=False,wall_color="r")
        ax = plot_y_wall(ax, wall_y2, wall_x1, wall_x2, boundary=False,wall_color="r")
        ax = plot_x_wall(ax, wall_x2, wall_y1, wall_y2, boundary=False,wall_color="r")
        # ax.axvline(x=x1,ymax=wall_yMax, ymin=wall_yMin, color = "b", lw = 2)
        # ax.axvline(x=x2,ymax=wall_yMax, ymin=wall_yMin, color = "b", lw = 2)
    # ax.set_title(f"Vicsek Model in Python. $\\rho = {rho}$, $\\eta = {eta}$")
    qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
    if wall_y2 - wall_y1 == L:
        ax.legend(loc = "upper right")
    ax.set_aspect("equal")
    ax.set_xlabel(r"x ($R_0$)")
    ax.set_ylabel(r"y ($R_0$)")
    ax.set_yticks(np.linspace(0,L,5))
    ax.set_xticks(np.linspace(0,L,5))
    ax.set_ylim(0,L)
    ax.set_xlim(0,L)
    fig.savefig(f"figures/snapshot_{(wall_y2-wall_y1)/L:.2f}_.png")

