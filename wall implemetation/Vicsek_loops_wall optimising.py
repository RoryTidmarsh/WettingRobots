import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
# parameters
L = 64 # size of box
rho = 1 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.1 # noise/randomness
max_num_neighbours= N


#Defining parameters for a wall only in the x direction.
wall_x = L/2
wall_yMin = L/2 - L/4
wall_yMax = L/2 + L/4
wall_distance = r0/3
wall_turn = np.deg2rad(110)
turn_factor = 0.2

#Defining parameters for a rectangle
x_min, x_max, y_min, y_max = 4.,6.,4.,6.

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

step_num = 0
# For alignment Graph
av_frames_angles = 10
num_frames_av_angles = np.empty(av_frames_angles)
t = 0
@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))
average_angles = [average_angle(positions)]


###Average position in a 2D histogram over time
bins = int(L*5/r0)
hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= bins, density = False)

### Streamplot setup
old_pos = positions.copy()
nbins=64
bin_edges = np.linspace(0,L,nbins) 
centres = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0]) # Centers for streamplot
X,Y = np.meshgrid(centres,centres) #meshgrid for streamplot
dr = positions-old_pos  # Change _streamin pos_streamiiton
_Hx_stream, edgex_stream,edgey_stream = np.histogram2d(old_pos[:,0],old_pos[:,1],weights=dr[:,0], bins=(bin_edges,bin_edges)) #initialising the histograms
_Hy_stream,edgex_stream,edgey_stream = np.histogram2d(old_pos[:,0],old_pos[:,1],weights=dr[:,1], bins=(bin_edges,bin_edges))


#### Wall funcitons ####
@numba.njit
def x_wall_filter(x_pos,y_pos):
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
def varying_angle_turn(x_pos, y_pos, turn_factor):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        complex_turn: complex number representing the angle
    """
    x_dist = x_pos- wall_x
    
    dist = x_wall_filter(x_pos, y_pos)
    if dist < wall_distance:
        if wall_yMin <= y_pos <= wall_yMax:
            y_dist = 0 # Turn off any interaction in y direction
            
        elif wall_yMin > y_pos:
            beta = 1
            y_dist =  y_pos - wall_yMin
        elif wall_yMax < y_pos:
            y_dist = y_pos - wall_yMax

        r_hat = (x_dist +y_dist*1j)/(x_dist**2 +y_dist**2)
    
        #complex_turn = alpha/(turn_factor*dist*r_hat.real) +beta/(turn_factor*r_hat.imag)*1.0j
        complex_turn = (turn_factor/dist)*r_hat
    else:
        complex_turn = 0
    return complex_turn

def plot_x_wall(ax, wall_color = "blue", boundary = True):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
    if boundary==True:
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
    ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall", color = wall_color)
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

positions = position_filter(positions, x_wall_filter) # No particles spawn in turn boundary

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
                        distance_sq = dx * dx + dy * dy
                        if distance_sq < r0 * r0:  # Compare with squared radius
                            if count_neigh < max_num_neighbours:
                                neigh_angles[count_neigh] = angles[j]
                                count_neigh += 1

        x_pos = positions[i,0]
        y_pos = positions[i,1]
          
        # Calculate the angle to turn due to the wall
        wall_turn = varying_angle_turn(x_pos, y_pos,turn_factor=turn_factor)
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

def animate(frames):
    print(frames)
    global positions, angles
    
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell)
    
    global t, num_frames_av_angles
    # Store the new angles in the num_frames_av_angles array
    num_frames_av_angles[t] = average_angle(new_angles)
    if t == av_frames_angles - 1:  # Check if we've filled the array
        average_angles.append(average_angle(num_frames_av_angles))
        t = 0  # Reset t
        num_frames_av_angles = np.empty(av_frames_angles)  # Reinitialize the array
    else:
        t += 1  # Increment t
        
    global hist_pos, step_num
    #Add positions to the 2D histogram for position
    hist_pos += np.histogram2d(new_positions[:, 0], new_positions[:,1], bins= [xedges,yedges], density = False)[0]

    global _Hx_stream, _Hy_stream
    #Add change in position to the 2D histograms for streamplot
    dr = new_positions-positions  # Change in position
    dr = np.where(dr >5.0, dr-10, dr)
    dr = np.where(dr < -5.0, dr+10, dr) #Filtering to see where the paricles go over the periodic boundary conditions
    H_stream,edgex_stream,edgey_stream = np.histogram2d(old_pos[:,0],old_pos[:,1],weights=dr[:,0], bins=(bin_edges,bin_edges))  # dr_x wieghted histogram
    _Hx_stream += H_stream
    H_stream,edgex_stream,edgey_stream = np.histogram2d(old_pos[:,0],old_pos[:,1],weights=dr[:,1], bins=(bin_edges,bin_edges))  # dr_y wieghted histogram
    _Hy_stream +=H_stream


    # Update global variables
    positions = new_positions.copy()
    angles = new_angles.copy()
    step_num +=1

    ###Update the quiver plot  Comment out up to and inculding the return statement if you do not whish to have the animation
    # qv.set_offsets(positions)
    # qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    # return qv,
 
### Showing the animation
# fig, ax = plt.subplots(figsize = (6, 6))   
# ax = plot_x_wall(ax, boundary = False)
# ax.set_title(f"Viscek {N} particles, eta = {eta} .")
# qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# ani = FuncAnimation(fig, animate, frames = range(1, int(iterations/10)), interval = 5, blit = True)
# ax.legend(loc = "upper right")
# # ani.save(f'figures/Vicek_={rho}_eta={eta}.gif', writer='pillow', fps=30)
# plt.show()


wall_colour = "r"
# #### STREAM PLOT ##### (currently works on a for loop without the animation)
fig, ax = plt.subplots(figsize = (12, 6), ncols = 2)   
ax[0] = plot_x_wall(ax[0], wall_color = wall_colour,boundary = False)
ax[0].set_title(f"Viscek {N} particles, eta = {eta} .")

####Uncomment these next 3 lines to run the simulation without animating
nsteps = 3000
for i in range(1, nsteps+1):   # Running the simulaiton
    animate(i)

## STREAM PLOT
_Hx_stream/=(step_num) # sNormailising
_Hy_stream/=(step_num)

# print(X.shape,_Hx_stream.shape)
ax[0].streamplot(X,Y,_Hx_stream,_Hy_stream)   


## 2D HISTOGRAM
hist_normalised = hist_pos.T/sum(hist_pos)
# Use imshow to display the normalized histogram
cax = ax[1].imshow(hist_normalised, extent=[0, L, 0, L], origin='lower', cmap='cividis', aspect='auto')
ax[1] = plot_x_wall(ax[1], wall_color = wall_colour, boundary= False)
ax[1].set_xlabel("X Position")
ax[1].set_ylabel("Y Position")
ax[1].set_title(f"2D Histogram of Particle Positions over {step_num} timesteps.")
ax[1].legend()
# Add a colorbar for reference
fig.colorbar(cax, ax=ax[1], label='Density')

plt.show()

# fig, ax2 = plt.subplots()
# times = np.arange(0,len(average_angles))*av_frames_angles
# ax2.plot(times, average_angles, label = "10 frame average")
# # ax2.plot(np.arange(len(average_angles2)), average_angles2, label = "1 frame")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Angle (radians)")
# ax2.set_title("Alignment, averaging from different number of frames.")
# ax2.legend(loc = "upper left")

# ax2.plot([0,times.max()],[-np.pi, -np.pi], linestyle = "--", color = "grey", alpha = 0.4)
# ax2.plot([0,times.max()],[np.pi, np.pi], linestyle = "--", color = "grey", alpha = 0.4)
# ax2.grid()
# # plt.show()

# hist_normalised = hist.T/sum(hist)
# # After the animation and histogram calculations
# fig, ax3 = plt.subplots(figsize=(6, 6))
# # Use imshow to display the normalized histogram
# cax = ax3.imshow(hist_normalised, extent=[0, L, 0, L], origin='lower', cmap='rainbow', aspect='auto')
# ax3 = plot_x_wall_boundary(ax3, "red")
# ax3.set_xlabel("X Position")
# ax3.set_ylabel("Y Position")
# ax3.set_title(f"2D Histogram of Particle Positions over {len(times)*10} timesteps.")
# ax3.legend()
# # Add a colorbar for reference
# fig.colorbar(cax, ax=ax3, label='Density')
# plt.show()