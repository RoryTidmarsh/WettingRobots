"""animation of Vicsek model with walls. Currently in a reduced system to increase the run speed

There are no outputs in this file
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
from pycircular.stats import periodic_mean_std
import os

# parameters
L = 64 # size of box
rho = 1 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.1  # noise/randomness
max_num_neighbours= 100


# Curved wall parameters
radius = L/2    # radius of the circle
center_x = L/2 #- radius  # x-coordinate of circle center
center_y = L/2  # y-coordinate of circle center
arc_angle = 4*np.pi/2  # length of arc in radians (pi/2 = quarter circle)
start_angle = -arc_angle/2  # starting angle of the arc
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
def point_to_arc_distance(x,y, center_x, center_y, radius, start_angle, arc_angle):
    """Calculate the distance from a point to an arc.

    Args:
        x (float): x coordinate of particle
        y (float): y coordinate of particle
        center_x (float): central x point of arc
        center_y (float): central y point of arc
        radius (float): radius of arc
        start_angle (float): angular position of the start of the curve (radians)
        arc_angle (float): total angle taken for the curve

    Returns:
        dist, r_hat(tuple of floats): dist = distance to the wall, r_hat (comple number) is the unitvector from the arc to the particle
    """

    # Vector distance to center point
    dx = x - center_x
    dy = y - center_y

    # Angle from the center to the point
    angle_from_center = np.arctan2(dy,dx)

    # Calculating the x,y of the end of the arc
    end_angle = start_angle + arc_angle
    end1_x = radius*np.cos(start_angle)
    end1_y = radius*np.sin(start_angle)
    end2_x = radius*np.cos(end_angle)
    end2_y = radius*np.sin(end_angle)
    # Check the particle is within the range wanted
    if start_angle <= angle_from_center <= end_angle: 
        disp = np.sqrt(dx*dx + dy*dy) - radius # radial displacement to the curve
        r_hat = (np.cos(angle_from_center) + 1j*np.sin(angle_from_center))*np.sign(disp) # direction from curve to point
        return np.absolute(disp), r_hat
    
    elif angle_from_center>= end_angle:    # Angle greater than the end angle
        # Calculate the distance from end2 to point
        dx = x - end2_x
        dy = y - end2_y
        dist = np.sqrt(dx*dx + dy*dy) # distance from end
        r_hat = (dx + 1j*dy)/dist # direction to curve
        return dist, r_hat
    elif angle_from_center<= start_angle:    # Angle greater than the end angle
        # Calculate the distance from end2 to point
        dx = x - end1_x
        dy = y - end1_y
        dist = np.sqrt(dx*dx + dy*dy) # distance from end
        r_hat = (dx + 1j*dy)/dist # direction to curve
        return dist, r_hat


@numba.njit
def arc_angle_turn(x_pos, y_pos, turn_factor):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        x_pos, y_pos (float): particle positions
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        complex_turn: complex number representing the angle
    """
    dist, r_hat = point_to_arc_distance(x_pos, y_pos, center_x,center_y, radius, start_angle, arc_angle)
    if dist < wall_distance:    
        complex_turn = (turn_factor/dist)*r_hat
    else:
        complex_turn = 0
    return complex_turn

def plot_arc(ax, wall_color = "blue", boundary = True, walpha = 1):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
    global center_x,center_y, start_angle, radius, arc_angle, wall_distance
    thetas = np.linspace(start_angle, start_angle+arc_angle,200)
    x = center_x + radius*np.cos(thetas)
    y = center_y + radius*np.sin(thetas)
    ax.plot(x,y, color = wall_color, alpha= walpha)

    if boundary==True:
        #Plotting the far and near arcs
        x_far = center_x + (radius+wall_distance)*np.cos(thetas)
        y_far = center_y + (radius+wall_distance)*np.sin(thetas)
        ax.plot(x_far,y_far, 'b--', lw=2)

        x_close = center_x + (radius-wall_distance)*np.cos(thetas)
        y_close = center_y + (radius-wall_distance)*np.sin(thetas)
        ax.plot(x_close,y_close, 'b--', lw=2)

        # Plot full circles at each endpoint
        end1_x = center_x + radius * np.cos(start_angle)
        end1_y = center_y + radius * np.sin(start_angle)
        end2_x = center_x + radius * np.cos(start_angle + arc_angle)
        end2_y = center_y + radius * np.sin(start_angle + arc_angle)
        circle_thetas = np.linspace(0, 2*np.pi, 100)
        # First endpoint circle
        ax.plot(end1_x + wall_distance * np.cos(circle_thetas),
                end1_y + wall_distance * np.sin(circle_thetas),
                'b--', lw=2)
        # Second endpoint circle
        ax.plot(end2_x + wall_distance * np.cos(circle_thetas),
                end2_y + wall_distance * np.sin(circle_thetas),
                'b--', lw=2)
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
def update(positions, angles, cell_size, num_cells, max_particles_per_cell, wall_yMax, wall_yMin):
    
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
                        distance_sq = dx * dx + dy * dy
                        if distance_sq < r0 * r0:  # Compare with squared radius
                            if count_neigh < max_num_neighbours:
                                neigh_angles[count_neigh] = angles[j]
                                count_neigh += 1

        x_pos = positions[i,0]
        y_pos = positions[i,1]
          
        # Calculate the angle to turn due to the wall
        # wall_turn = varying_angle_turn(x_pos, y_pos,turn_factor=turn_factor, wall_yMin=wall_yMin, wall_yMax=wall_yMax)
        if arc_angle ==0:   # no wall case
            wall_turn = 0
        else:
            wall_turn = arc_angle_turn(x_pos, y_pos, turn_factor)
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

def animate(frames, wall_yMax, wall_yMin):
    # print(frames)
    global positions, angles, step_num
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell, wall_yMax, wall_yMin)


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
# ax = plot_x_wall(ax, boundary = False)
ax = plot_arc(ax, boundary=True)
ax.set_title(f"Vicsek Model in Python. $\\rho = {rho}$, $\\eta = {eta}$")
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# Add a color bar
cbar = fig.colorbar(qv, ax=ax, label="Angle (radians)")
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

ani = FuncAnimation(fig, animate, frames = range(1, int(iterations/10)), interval = 5, blit = True, fargs = (0,0))
ax.legend(loc = "upper right")
# ani.save(f'figures/Vicsek_={rho}_eta={eta}.gif', writer='pillow', fps=30)
plt.show()
# np.savez_compressed(f'{os.path.dirname(__file__)}/wall_size_experiment/finalstate.npz', Positions = positions, Orientation = angles)

