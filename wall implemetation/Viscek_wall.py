"""
File that simulates Vicsek model with the walls and outputs all data for analysis plots into .npz files. 

STATUS:
 - uncomment .npz lines to save. My simulations are saved under the folders beginning with "test128" 

to read the .npz files use analysis.py
"""
import numpy as np
import numba
from pycircular.stats import periodic_mean_std
import os
from tqdm import tqdm

# parameters
L = 64 # size of box
rho = 1 # density
N = int(rho * L**2) # number of particles
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0.1  # noise/randomness
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
positions = np.random.uniform(0, L, size=(N, 2))
angles = np.random.uniform(-np.pi, np.pi, size=N)

# Ensure all particles are spawned within the barriers
confined = True
if confined:
    positions = np.random.uniform([wall_x1, wall_y1], [wall_x2, wall_y2], size=(N, 2))

#### INITIALISATION OF DATA STORAGE ARAYS - for creation of analysis figures ####
# For alignment Graph
av_frames_angles = 10
num_frames_av_angles = np.empty(av_frames_angles)
num_frames_std_angles = np.empty(av_frames_angles)
t = 0
@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))    #or just use pycircular module
def average_orientation(new_angles):
    return np.absolute(np.sum(np.exp(new_angles*1.0j))/N)

av_angle, angle_std = periodic_mean_std(angles)
average_angles = [av_angle]
std_angles = [angle_std]
average_orientations = [average_orientation(angles)]


###Average position in a 2D histogram over time
hbins = int(L*5/r0)
hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= hbins, range=[[0,L],[0,L]], density = False)

### Streamplot setup
bins = int(L / r0)
tot_vx_all = np.zeros((bins, bins)) # velocity x components for all particles
tot_vy_all = np.zeros((bins, bins))
counts_all = np.zeros((bins, bins)) # number of particles in cell
vxedges = np.linspace(0, 1, bins + 1) # bin edges for meshgrid
vyedges = np.linspace(0, 1, bins + 1)
X,Y = np.meshgrid(vxedges[:-1],vyedges[:-1]) # meshgrid for quiver plot

#### Wall funcitons ####
@numba.njit
def x_wall_filter(x_pos,y_pos, wall_yMax, wall_yMin, wall_x=wall_x1):
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
    """Finds the distance of the particle to the wall in the y direction.

    Args:
        x_pos (float): x_position of 1 particle
        y_pos (float): y_position of 1 particle
    Returns:
        distance_to_wall: distance of the particle to the wall
    """
    if x_pos > wall_xMax:
        #particle to the right of the wall
        distance_to_wall = np.sqrt((x_pos-wall_xMax)**2 + (y_pos-wall_y)**2)
    elif x_pos < wall_xMin:
        #particle to the left of the wall
        distance_to_wall = np.sqrt((x_pos-wall_xMin)**2 + (y_pos-wall_y)**2)
    else:
        #particle level with the wall
        distance_to_wall = np.abs(y_pos-wall_y)
    return distance_to_wall

@numba.njit
def varying_angle_turn(x_pos, y_pos, turn_factor, wall_y1, wall_y2, wall_x):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        complex_turn: complex number representing the angle
    """
    x_dist = x_pos- wall_x
    
    dist = x_wall_filter(x_pos, y_pos, wall_y2, wall_y1, wall_x)
    if dist < wall_distance:
        if wall_y1 <= y_pos <= wall_y2:
            y_dist = 0 # Turn off any interaction in y direction
            
        elif wall_y1 > y_pos:
            y_dist =  y_pos - wall_y1
        elif wall_y2 < y_pos:
            y_dist = y_pos - wall_y2

        r_hat = (x_dist +y_dist*1j)/(x_dist**2 +y_dist**2)
    
        #complex_turn = alpha/(turn_factor*dist*r_hat.real) +beta/(turn_factor*r_hat.imag)*1.0j
        complex_turn = (turn_factor/dist)*r_hat
    else:
        complex_turn = 0
    return complex_turn

@numba.njit
def varying_angle_turn_y(x_pos, y_pos, turn_factor, wall_x1, wall_x2, wall_y):
    """Tells the particle how much to turn when it is within the boundary in the y direction

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        complex_turn: complex number representing the angle
    """
    y_dist = y_pos- wall_y
    
    dist = y_wall_filter(x_pos, y_pos, wall_x2, wall_x1, wall_y)
    if dist < wall_distance:
        if wall_x1 <= x_pos <= wall_x2:
            x_dist = 0 # Turn off any interaction in x direction
            
        elif wall_x1 > x_pos:
            x_dist =  x_pos - wall_x1
        elif wall_x2 < x_pos:
            x_dist = x_pos - wall_x2

        r_hat = (x_dist +y_dist*1j)/(x_dist**2 +y_dist**2)
    
        #complex_turn = alpha/(turn_factor*dist*r_hat.real) +beta/(turn_factor*r_hat.imag)*1.0j
        complex_turn = (turn_factor/dist)*r_hat
    else:
        complex_turn = 0
    return complex_turn

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
def update(positions, angles, cell_size, num_cells, max_particles_per_cell, wall_y1, wall_y2, wall_x1, wall_x2, eta = eta):
    
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
    global positions, angles, step_num,eta
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell, wall_y1, wall_y2, wall_x1, wall_x2,eta=eta)
    
    ### NEEDED FOR ANALYSIS FIGURES
    ## alignment graphs
    global t, num_frames_av_angles, num_frames_std_angles
    # Store the new angles in the num_frames_av_angles array
    num_frames_av_angles[t], num_frames_std_angles[t] = periodic_mean_std(new_angles)
    if t == av_frames_angles - 1:  # Check if we've filled the array
        angle_mean, _ = periodic_mean_std(num_frames_av_angles)
        angle_std, _ = periodic_mean_std(num_frames_std_angles)
        average_angles.append(angle_mean)
        std_angles.append(angle_std)
        t = 0  # Reset t
        num_frames_av_angles = np.empty(av_frames_angles)  # Reinitialize the array
    else:
        t += 1  # Increment t

    # Complex orientation 
    global average_orientations
    average_orientations.append(average_orientation(angles))
        
    global hist_pos
    #Add positions to the 2D histogram for position
    hist_pos += np.histogram2d(new_positions[:, 0], new_positions[:,1], bins= [xedges,yedges], density = False)[0]

    global tot_vx_all, tot_vy_all, counts_all, vxedges, vyedges 
    #Add change in position to the 2D histograms for streamplot
    dr = new_positions-positions  # Change in position
    dr = np.where(dr >5.0, dr-10, dr)
    dr = np.where(dr < -5.0, dr+10, dr) #Filtering to see where the paricles go over the periodic boundary conditions
    # histograms for the x and y velocity components
    H_vx, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, range=[[0,L],[0,L]], weights = dr[:,0])
    H_vy, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, range=[[0,L],[0,L]], weights = dr[:,1])
    counts, vxedges, vyedges = np.histogram2d(positions[:,0], positions[:,1], bins = bins, range=[[0,L],[0,L]])
    tot_vx_all += H_vx
    tot_vy_all += H_vy
    counts_all += counts # hist of number of particles
    
    # Update global variables
    positions = new_positions.copy()
    angles = new_angles.copy()
    step_num +=1


### SAVING DATA AS .npz FILES ###
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



# ####Uncomment these next lines to run the simulation without animating
nsteps = 10000
current_dir = os.path.dirname(__file__)

start_l_ratio = 0.75
start_eta = 0
start_J = 0
# Loop for each wall length
for l_ratio in [0.75]:#np.linspace(0,1,6)[1::2]:
    # J=0
    # Initialising the new wall
    l_ratio = float(l_ratio)
    l = L* l_ratio

    # Confined square geometry
    wall_yMin = L / 2 - l / 2
    wall_yMax = L / 2 + l / 2
    wall_xMin = L / 2 - l / 2
    wall_xMax = L / 2 + l / 2

    wall_y1 = wall_yMin
    wall_y2 = wall_yMax
    wall_x1 = wall_xMin
    wall_x2 = wall_xMax

    
# Creating a directory for this wallsize to fall into
    exp_dir = [f"/wall_size_experiment/{int(L)}L", "/noise_experiment", "/wall_size_experiment/50wall"]
    savedir = current_dir + exp_dir[0] + f"/square_{l}_{nsteps}"
    # delete_files_in_directory(savedir)
    os.makedirs(savedir, exist_ok=True)
    output_parameters(savedir)

    if l_ratio < start_l_ratio:
        continue

    for eta in [0.1]:
        if eta < start_eta and l_ratio <= start_l_ratio:
            continue
        for J in range(0,4):
            if (l_ratio <= start_l_ratio and eta <= start_eta) and J < start_J:
                continue
            # initialise positions and angles for the new situation
            positions = np.random.uniform([wall_x1, wall_y1], [wall_x2, wall_y2], size=(N, 2))
            angles = np.random.uniform(-np.pi, np.pi, size = N) 

            # Creating the inital storage for each plot
            # Density Histogram
            hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= hbins, range=[[0,L],[0,L]], density = False) 
            ## Direction and Spread plots
            av_angle, angle_std = periodic_mean_std(angles) # Average angle of inital setup
            average_angles = [av_angle] # Create arrays witht the initial angles in s
            average_orientations = [average_orientation(angles)]
            std_angles = [angle_std]
            ### Streamplot setup
            bins = int(L / r0)
            tot_vx_all = np.zeros((bins, bins)) # velocity x components for all particles
            tot_vy_all = np.zeros((bins, bins))
            counts_all = np.zeros((bins, bins)) # number of particles in cell
            vxedges = np.linspace(0, 1, bins + 1) # bin edges for meshgrid
            vyedges = np.linspace(0, 1, bins + 1)
            X,Y = np.meshgrid(vxedges[:-1],vyedges[:-1]) # meshgrid for quiver plot
            

            if nsteps < 3000:
                transient_cutoff = nsteps
            else:
                transient_cutoff = 3000
            # Run the simulation
            for i in tqdm(range(1, nsteps+1), desc=f"Wall length ratio {l_ratio:.2f}, noise {eta}, Itt {J}"):
                animate(i, wall_y1, wall_y2, wall_x1, wall_x2)

                # store all the data from the transient phase            
                if i==transient_cutoff:
                    transient_hist_pos = hist_pos.copy()
                    transient_tot_vx_all = np.zeros_like(tot_vx_all)
                    transient_tot_vy_all = np.zeros_like(tot_vy_all)
                    transient_counts_all = np.zeros_like(counts_all)


                # reset the data for the steady state
                if i==transient_cutoff+1:
                    hist_pos = np.zeros_like(hist_pos)
                    tot_vx_all = np.zeros_like(tot_vx_all)
                    tot_vy_all = np.zeros_like(tot_vy_all)
                    counts_all = np.zeros_like(counts_all)

            # Saving into npz floats for later analysis
            np.savez_compressed(f'{savedir}/steady_histogram_data_{l:.2f}_{J}.npz', hist=np.array(hist_pos, dtype = np.float64))
            np.savez_compressed(f'{savedir}/transient_histogram_data_{l:.2f}_{J}.npz', hist=np.array(transient_hist_pos, dtype = np.float16))
            np.savez_compressed(f'{savedir}/steady_stream_plot_{l:.2f}_{J}.npz', X = X, Y= Y, X_hist = tot_vx_all, Y_hist = tot_vy_all, counts = counts_all)
            np.savez_compressed(f'{savedir}/transient_stream_plot_{l:.2f}_{J}.npz', X = X, Y= Y, X_hist = transient_tot_vx_all, Y_hist = transient_tot_vy_all, counts = transient_counts_all)
            np.savez_compressed(f'{savedir}/alignment_{l:.2f}_{J}.npz', angles = average_angles, std = std_angles)

            # Noise experiment
            np.savez_compressed(f'{savedir}/orientations_{l:.2f}_{J}.npz', orientations = average_orientations, noise = eta) 

            ## Name change of histogram for varying eta #### FOR NOISE CHANGE EXP ####
            # np.savez_compressed(f'{savedir}/steady_histogram_data_{eta}_{J}.npz', hist=np.array(hist_pos, dtype = np.float64))
            # np.savez_compressed(f'{savedir}/transient_histogram_data_{eta}_{J}.npz', hist=np.array(transient_hist_pos, dtype = np.float16))

            ## Saving positions and orientations for setup for recreation of the system
            # np.savez_compressed(f'{savedir}/finalstate_{eta}_{J}.npz', Positions = positions, Orientation = angles)
            
            # Reset the data storage arrays
            hist_pos = np.zeros_like(hist_pos)
            average_angles = []
            std_angles = []
            tot_vx_all = np.zeros_like(tot_vx_all)
            tot_vy_all = np.zeros_like(tot_vy_all)
            counts_all = np.zeros_like(counts_all)            
            average_orientations = []
