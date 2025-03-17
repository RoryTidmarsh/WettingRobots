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
from fractions import Fraction
from tqdm import tqdm
import matplotlib.pyplot as plt

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


# Curved wall parameters
center_x = L/2  # x-coordinate of circle center
center_y = L/2  # y-coordinate of circle center
radius = L/3    # radius of the circle
arc_angle = np.pi/2  # length of arc in radians (pi/2 = quarter circle)
start_angle = -arc_angle/2  # starting angle of the arc
wall_distance = 2*r0  # interaction distance from wall
turn_factor = 0.2

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

#### INITIALISATION OF DATA STORAGE ARAYS - for creation of analysis figures ####
# For alignment Graph
step_num = 0
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
hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= hbins, density = False)

### Streamplot setup
bins = int(L / r0)
tot_vx_all = np.zeros((bins, bins)) # velocity x components for all particles
tot_vy_all = np.zeros((bins, bins))
counts_all = np.zeros((bins, bins)) # number of particles in cell
vxedges = np.linspace(0, 1, bins + 1) # bin edges for meshgrid
vyedges = np.linspace(0, 1, bins + 1)
X, Y = np.meshgrid(vxedges[:-1] + (vxedges[1] - vxedges[0]) / 2, vyedges[:-1] + (vyedges[1] - vyedges[0]) / 2) # meshgrid for quiver plot

@numba.njit
def velocity_flux(positions, angles, v0):
    tot_vx = np.zeros(positions.shape[0]) # empty array for total velocity x components
    tot_vy = np.zeros(positions.shape[0]) # empty array for total velocity y components
    # calculate x and y velocity flux
    for i in numba.prange(positions.shape[0]):
        tot_vx[i] = v0 * np.cos(angles[i])
        tot_vy[i] = v0 * np.sin(angles[i])
    return tot_vx, tot_vy

#### Arc funcitons ####
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
def arc_angle_turn(x_pos, y_pos, turn_factor, arc_angle,start_angle):
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
def update(positions, angles, cell_size, num_cells, max_particles_per_cell, arc_angle, start_angle):
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
        if arc_angle ==0:   # no wall case
            wall_turn = 0
        else:
            wall_turn = arc_angle_turn(x_pos, y_pos, turn_factor, arc_angle, start_angle)
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

def animate(frames, arc_angle, start_angle=-arc_angle/2):
    global positions, angles, step_num
    new_positions, new_angles = update(positions, angles,cell_size, lateral_num_cells, max_particles_per_cell, arc_angle, start_angle)
    
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
    tot_vx, tot_vy = velocity_flux(positions, angles, v0)
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
#     #### Animation - Uncomment up to the next "###" to see the animation ###
#     #Update the quiver plot  
#     qv.set_offsets(positions)
#     qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
#     return qv,
 
# ## Showing the animation
# figwidth = 5.5
# totalheight = 4.675
# fig, ax = plt.subplots(figsize = (figwidth, totalheight))   
# # ax = plot_x_wall(ax, boundary = False)
# ax.set_title(f"Viscek Model in Python. $\\rho = {rho}$, $\\eta = {eta}$")
# qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
# # Add a color bar
# cbar = fig.colorbar(qv, ax=ax, label="Angle (radians)")
# cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
# cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

# ani = FuncAnimation(fig, animate, frames = range(1, int(iterations/10)), interval = 5, blit = True, fargs = (0,0))
# ax.legend(loc = "upper right")
# # ani.save(f'figures/Vicek_={rho}_eta={eta}.gif', writer='pillow', fps=30)
# plt.show()
# np.savez_compressed(f'{os.path.dirname(__file__)}/wall_size_experiment/finalstate.npz', Positions = positions, Orientation = angles)

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
    global arc_angle, radius, start_angle, center_x, center_y
    filename = file_dir + f'/simulation_parameters_{arc_angle}.txt'
    with open(filename, 'w') as f:
        f.write(f"Size of box (L): {L}\n")
        f.write(f"Density (rho): {rho}\n")
        f.write(f"Number of particles (N): {N}\n")
        f.write(f"Interaction radius (r0): {r0}\n")
        # f.write(f"Time step (deltat): {deltat}\n")
        # f.write(f"Velocity (v0): {v0}\n")
        f.write(f"Noise/randomness (eta): {eta}\n")
        f.write(f"Max number of neighbours: {max_num_neighbours}\n")
        f.write(f"Total number of step`s: {nsteps} \n")
        f.write(f"Arc_angle: {arc_angle} \n")
        f.write(f"Start angle: {start_angle} \n")
        f.write(f"Radius: {radius} \n")
        f.write(f"center_x: {center_x} \n")
        f.write(f"center_y: {center_y} \n")
        f.write(f"Alignment average steps: {av_frames_angles} \n")



# ####Uncomment these next lines to run the simulation without animating
nsteps = 10000
current_dir = os.path.dirname(__file__)

# Skipping to the relevant angle and iteration that the simulation stopped at
skip_angle = 2*np.pi
iteration_skip = 0
# # Loop for each wall length
for arc_angle in np.array([0,1/3,2/3,1,1.5, 2])*np.pi:#np.linspace(0,np.pi,4)[2:]:
    arc_angle = float(arc_angle)
    start_angle = -arc_angle/2
    # Creating a directory for this wallsize to fall into
    exp_dir = ["/wall_size_experiment", "/noise_experiment", "/arc_analysis"]
    savedir = current_dir + exp_dir[-1] + f"/arc{int(L)}_{arc_angle/np.pi:.2f}_{nsteps}"
    # delete_files_in_directory(savedir)
    os.makedirs(savedir, exist_ok=True)
    output_parameters(savedir)
    
    #skip to skip angle
    if arc_angle < skip_angle:
        continue
    # # Creating multiple iterations to be averaged for the alignment
    for J in range(4):
        if J < iteration_skip and arc_angle == skip_angle:
            continue
        # initialise positions and angles for the new situation
        positions = np.random.uniform(0, L, size = (N, 2))
        angles = np.random.uniform(-np.pi, np.pi, size = N) 

        if arc_angle == 2*np.pi:
            # Reset all positions to inside the circle
            for k in range(N):
                r_k = np.sqrt((positions[k,0]-center_x)**2 + (positions[k,1]-center_y)**2)
                while r_k >= radius:
                    positions[k] = np.random.uniform(0, L, size = 2)
                    r_k = np.sqrt((positions[k,0]-center_x)**2 + (positions[k,1]-center_y)**2)

        # Initialise histogram over the entire system
        hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=hbins, range=[[0, L], [0, L]], density=False)
        bin_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

        # Streamplot setup
        bins = int(L / r0)
        tot_vx_all = np.zeros((bins, bins)) # velocity x components for all particles
        tot_vy_all = np.zeros((bins, bins))
        counts_all = np.zeros((bins, bins)) # number of particles in cell
        vxedges = np.linspace(0, L, bins + 1) # bin edges for meshgrid
        vyedges = np.linspace(0, L, bins + 1)
        X, Y = np.meshgrid(vxedges[:-1], vyedges[:-1]) # meshgrid for quiver plot

        # Density Histogram
        # hist_pos, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= hbins, density = False) 
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
        for i in tqdm(range(1, nsteps+1), desc=f"Arc angle {Fraction(arc_angle/np.pi).limit_denominator(np.pi)}pi, Iteration {J}"):
            animate(i, arc_angle,start_angle)

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

        ## Saving into npz floats for later analysis
        np.savez_compressed(f'{savedir}/steady_histogram_data_{arc_angle/np.pi:.2f}_{J}.npz', hist=np.array(hist_pos, dtype = np.float64))
        np.savez_compressed(f'{savedir}/transient_histogram_data_{arc_angle/np.pi:.2f}_{J}.npz', hist=np.array(transient_hist_pos, dtype = np.float16))

        # New streamplot saving
        np.savez_compressed(f'{savedir}/steady_stream_plot_{arc_angle/np.pi:.2f}_{J}.npz', X = X, Y= Y, X_hist = tot_vx_all, Y_hist = tot_vy_all,counts = counts_all)
        np.savez_compressed(f'{savedir}/transient_stream_plot_{arc_angle/np.pi:.2f}_{J}.npz', X = X, Y= Y, X_hist = transient_tot_vx_all, Y_hist = transient_tot_vy_all, counts = transient_counts_all)
        # Direction and spread
        np.savez_compressed(f'{savedir}/alignment_{arc_angle/np.pi:.2f}_{J}.npz', angles = average_angles, std = std_angles)
        np.savez_compressed(f'{savedir}/orientations_{arc_angle/np.pi:.2f}_{J}.npz', orientations = average_orientations) 
        ## Saving positions and orientations for setup for recreation of the system
        # np.savez_compressed(f'{savedir}/finalstate_{arc_angle/np.pi:.2f}_{J}.npz', Positions = positions, Orientation = angles)

        fig, ax = plt.subplots()
        cax = ax.imshow(hist_pos.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax.streamplot(X*L, Y*L, tot_vy_all, tot_vx_all, color="white")
        ax.set_title(f"Stream Plot for arc_angle={arc_angle/np.pi:.2f}π, Iteration {J}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        cbar = fig.colorbar(cax, ax=ax)

        arc = plt.Circle((center_x, center_y), radius, color='r', fill=False, linestyle='--')
        ax.add_artist(arc)
        plt.show()

        # theta = np.linspace(start_angle, start_angle + arc_angle, 100)
        # x_arc = center_x + radius * np.cos(theta)
        # y_arc = center_y + radius * np.sin(theta)
        # ax.plot(x_arc, y_arc, 'r--')

        # fig1, ax1 = plt.subplots()
        # ax1.quiver(positions[:, 0], positions[:, 1], np.cos(angles), np.sin(angles), angles, clim=[-np.pi, np.pi], cmap="hsv")
        # ax1.set_title(f"Final State of the System for arc_angle={arc_angle/np.pi:.2f}π, Iteration {J}")
        # plt.show()
        # Reset the data storage arrays
        hist_pos = np.zeros_like(hist_pos)
        tot_vx_all = np.zeros_like(tot_vx_all)
        tot_vy_all = np.zeros_like(tot_vy_all)
        counts_all = np.zeros_like(counts_all)
        average_angles = []
        average_orientations = []
        std_angles = []