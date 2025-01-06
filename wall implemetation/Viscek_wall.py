import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
from pycircular.stats import periodic_mean_std
import os

# parameters
L = 128 # size of box
rho = 1 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.1  # noise/randomness
max_num_neighbours= 100


#Defining parameters for a wall only in the x direction.
wall_x = L/2
wall_yMin = 0   #L/2 - L/4
wall_yMax = L   #L/2 + L/4
wall_distance = r0/3
wall_turn = np.deg2rad(110)
turn_factor = 0.2

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

step_num = 0
# For alignment Graph
av_frames_angles = 10
num_frames_av_angles = np.empty(av_frames_angles)
num_frames_std_angles = np.empty(av_frames_angles)
t = 0
@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))

av_angle, angle_std = periodic_mean_std(angles)
average_angles = [av_angle]
std_angles = [angle_std]


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
def x_wall_filter(x_pos,y_pos, wall_yMax, wall_yMin):
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
def varying_angle_turn(x_pos, y_pos, turn_factor, wall_yMax, wall_yMin):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        complex_turn: complex number representing the angle
    """
    x_dist = x_pos- wall_x
    
    dist = x_wall_filter(x_pos, y_pos, wall_yMax, wall_yMin)
    if dist < wall_distance:
        if wall_yMin <= y_pos <= wall_yMax:
            y_dist = 0 # Turn off any interaction in y direction
            
        elif wall_yMin > y_pos:
            y_dist =  y_pos - wall_yMin
        elif wall_yMax < y_pos:
            y_dist = y_pos - wall_yMax

        r_hat = (x_dist +y_dist*1j)/(x_dist**2 +y_dist**2)
    
        #complex_turn = alpha/(turn_factor*dist*r_hat.real) +beta/(turn_factor*r_hat.imag)*1.0j
        complex_turn = (turn_factor/dist)*r_hat
    else:
        complex_turn = 0
    return complex_turn

def plot_x_wall(ax, wall_color = "blue", boundary = True, walpha = 1):
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
    ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall", color = wall_color, alpha = walpha)
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
        if wall_yMin == wall_yMax:
            # i.e. there is no wall
            wall_turn = 0
        else:
            wall_turn = varying_angle_turn(x_pos, y_pos,turn_factor=turn_factor, wall_yMin=wall_yMin, wall_yMax=wall_yMax)
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
    print(frames)
    global positions, angles, step_num
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell, wall_yMax, wall_yMin)
    
    ## NEEDED FOR ANALYSIS FIGURES IN THIS FILE
    # ## alignment graphs
    # global t, num_frames_av_angles, num_frames_std_angles
    # # Store the new angles in the num_frames_av_angles array
    # num_frames_av_angles[t], num_frames_std_angles[t] = periodic_mean_std(new_angles)
    # if t == av_frames_angles - 1:  # Check if we've filled the array
    #     angle_mean, _ = periodic_mean_std(num_frames_av_angles)
    #     angle_std, _ = periodic_mean_std(num_frames_std_angles)
    #     average_angles.append(angle_mean)
    #     std_angles.append(angle_std)
    #     t = 0  # Reset t
    #     num_frames_av_angles = np.empty(av_frames_angles)  # Reinitialize the array
    # else:
    #     t += 1  # Increment t
        
    global hist_pos
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
    
#     #Update the quiver plot  Comment out up to and inculding the return statement if you do not whish to have the animation
#     qv.set_offsets(positions)
#     qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
#     return qv,
 
# ## Showing the animation
# fig, ax = plt.subplots(figsize = (7, 6))   
# ax = plot_x_wall(ax, boundary = False)
# ax.set_title(f"Viscek Model in Python. $\\rho = {rho}$, $\\eta = {eta}$")
# qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# # Add a color bar
# cbar = fig.colorbar(qv, ax=ax, label="Angle (radians)")
# cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
# cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
# ani = FuncAnimation(fig, animate, frames = range(1, int(iterations/10)), interval = 5, blit = True, fargs = (wall_yMax,wall_yMin))
# ax.legend(loc = "upper right")
# # ani.save(f'figures/Vicek_={rho}_eta={eta}.gif', writer='pillow', fps=30)
# plt.show()

### SAVING DATA AS .npz FILES
#finding the current working directory
current_dir = os.getcwd()
def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

# Function to output parameters to a file
def output_parameters(file_dir):
    """Creates a descriptive file of the data in the directory

    Args:
        filedir (str): the directory of the file location 
    """
    global l
    filename = file_dir + f'/simulation_parameters_{l}.txt'
    with open(filename, 'w') as f:
        f.write(f"Size of box (L): {L}\n")
        f.write(f"Density (rho): {rho}\n")
        f.write(f"Number of particles (N): {N}\n")
        f.write(f"Interaction radius (r0): {r0}\n")
        # f.write(f"Time step (deltat): {deltat}\n")
        # f.write(f"Velocity (v0): {v0}\n")
        f.write(f"Noise/randomness (eta): {eta}\n")
        f.write(f"Max number of neighbours: {max_num_neighbours}\n")
        f.write(f"Total number of steps: {nsteps} \n")
        f.write(f"Wall size (l): {l} \n")
        f.write(f"Alignment average steps: {av_frames_angles} \n")



####Uncomment these next lines to run the simulation without animating
nsteps = 10000
current_dir = os.path.dirname(__file__)


# Loop for each wall length
for l_ratio in np.linspace(0,1,6):
    # Initialising the new wall
    l = L* l_ratio
    wall_yMin = L/2 - l/2
    wall_yMax = L/2 + l/2

    # Creating a directory for this wallsize to fall into
    savedir = current_dir + "/wall_size_experiment" + f"/wall128_{l}"
    delete_files_in_directory(savedir)
    os.makedirs(savedir, exist_ok=True)
    output_parameters(savedir)

    # initialise positions and angles for the new situation
    positions = np.random.uniform(0, L, size = (N, 2))
    angles = np.random.uniform(-np.pi, np.pi, size = N) 

    # Creating the inital storage for each plot
    hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= bins, density = False) #Position histogram
    _Hx_stream, edgex_stream,edgey_stream = np.histogram2d(old_pos[:,0],old_pos[:,1],weights=dr[:,0], bins=(bin_edges,bin_edges)) # stream plot histogram
    _Hy_stream,edgex_stream,edgey_stream = np.histogram2d(old_pos[:,0],old_pos[:,1],weights=dr[:,1], bins=(bin_edges,bin_edges)) # stream plot histogram
    # av_angle, angle_std = periodic_mean_std(angles) # Average angle of inital setup
    # average_angles = [av_angle] # Create arrays witht the initial angles in s
    # std_angles = [angle_std]

#     # Run the simulation
    for i in range(1, nsteps+1):
        animate(i, wall_yMax, wall_yMin)

        # store all the data from the transient phase
        if i==3000:
            transient_hist_pos = hist_pos.copy()
            transient_Hx_stream = _Hx_stream.copy()
            transient_Hy_stream = _Hy_stream.copy()

        # reset the data for the steady state
        if i==5000:
            hist_pos = np.zeros_like(hist_pos)
            _Hx_stream = np.zeros_like(_Hx_stream)
            _Hy_stream = np.zeros_like(_Hy_stream)

    ## Saving into npz floats for later analysis
    np.savez_compressed(f'{savedir}/steady_histogram_data_{l}.npz', hist=np.array(hist_pos, dtype = np.float16))
    np.savez_compressed(f'{savedir}/transient_histogram_data_{l}.npz', hist=np.array(transient_hist_pos, dtype = np.float16))
    np.savez_compressed(f'{savedir}/steady_stream_plot_{l}.npz', X = X, Y= Y, X_hist = _Hx_stream, Y_hist = _Hy_stream)
    np.savez_compressed(f'{savedir}/transient_stream_plot_{l}.npz', X = X, Y= Y, X_hist = transient_Hx_stream, Y_hist = transient_Hy_stream)
    # np.savez_compressed(f'{savedir}/alignment_{l}.npz', angles = average_angles, std = std_angles)

    hist_pos = np.zeros_like(hist_pos)
    _Hx_stream = np.zeros_like(_Hx_stream)
    _Hy_stream = np.zeros_like(_Hy_stream)
    # average_angles = []
    # std_angles = []


# #### Code below is for in file creation of analysis plots, this can be done when saved using above files.
# #### STREAM PLOT ##### (currently works on a for loop without the animation)
# wall_colour = "r"
# fig, ax = plt.subplots(figsize = (12, 6), ncols = 2)   
# ax[0] = plot_x_wall(ax[0], wall_color = wall_colour,boundary = False, walpha= 0.5)
# ax[0].set_title(f"Viscek {N} particles, eta = {eta} .")

# _Hx_stream/=(step_num) # sNormailising
# _Hy_stream/=(step_num)

# # print(X.shape,_Hx_stream.shape)
# ax[0].streamplot(X,Y,_Hx_stream,_Hy_stream)   


# ## 2D HISTOGRAM
# hist_normalised = hist_pos.T/sum(hist_pos)
# # Use imshow to display the normalized histogram
# cax = ax[1].imshow(hist_normalised, extent=[0, L, 0, L], origin='lower', cmap='cividis', aspect='auto')
# ax[1] = plot_x_wall(ax[1], wall_color = '#FF00FF', boundary= False)
# ax[1].set_xlabel("X Position")
# ax[1].set_ylabel("Y Position")
# ax[1].set_title(f"2D Histogram of Particle Positions over {step_num} timesteps.")
# ax[1].legend()
# # Add a colorbar for reference
# fig.colorbar(cax, ax=ax[1], label='Density')
# fig.savefig(f"figures/Streamplot_Histogram_eta={eta}_rho={rho}_steps={nsteps}.png")
# plt.show()

### Alignment plot
# average_angles = np.array(average_angles)
# std_angles = np.array(std_angles)
# fig, ax2 = plt.subplots()
# times = np.arange(0,len(average_angles))*av_frames_angles
# ax2.plot(times, average_angles, label = "10 frame average")
# ax2.fill_between(times, average_angles - std_angles, average_angles + std_angles, alpha = 0.3, label = r"$\sigma$ of angles")
# # ax2.plot(np.arange(len(average_angles2)), average_angles2, label = "1 frame")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Angle (radians)")
# ax2.set_title(f"Average Alignment of {N} Particles, eta = {eta}")
# ax2.set_ylim(-3.22,3.22)
# ax2.legend()
# ax2.plot([0,times.max()],[-np.pi, -np.pi], linestyle = "--", color = "grey", alpha = 0.4) ## Lower angle limit
# ax2.plot([0,times.max()],[np.pi, np.pi], linestyle = "--", color = "grey", alpha = 0.4) ## Upper angle limit
# ax2.grid()
# fig.savefig(f"figures/Alginment_eta={eta}_rho={rho}_steps={nsteps}.png")
# plt.show()