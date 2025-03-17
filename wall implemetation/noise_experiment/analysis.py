import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "/wall implemetation/noise_experiment"
filenames = os.listdir(dir)
dir_starter = "DistanceNoise64"
# print(filenames)
cmap = "hsv"
L = 64
r0 = 1



histograms = True
timeav_noise = False
single_noise = False
histogram_filters = False # Initial attempt to find the distance from the wall
Density_profile = True # rho vs x for diiferent etas
Density_profile2 = True # rho vs x for diiferent lw
# save = False

text_width = 3.25  # inches (single column width)
fig_width = text_width
fig_height = 0.8 * fig_width  # for standard plots
width_2_subplot = fig_width/2 + 0.25  # for side-by-side subplots
height_2_subplot = 0.75 * width_2_subplot
height_cbar_2_subplot = 0.75 * width_2_subplot
plt.rcParams.update({
    'font.size': 10,
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
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    # "mathtext.fontset": "cm",
})

def read_summary_file(filepath):
    summary_data = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Split the line into key and value
            key, value = line.strip().split(': ')
            # Store the key-value pair in the dictionary
            summary_data[key] = float(value) if '.' in value or value.isdigit() else value
    return summary_data

wall_lengths = []
etas = []
def read_orientations():
    orientation_data = []
    global wall_lengths, etas,L,r0
    # Each folder contains data for each wall length
    for exp_item in filenames:
        if exp_item.split("_")[0] == dir_starter:
            folder_path = dir + "/" + exp_item
            wall_length = exp_item.split("_")[1]
            if wall_length not in wall_lengths:
                    wall_lengths.append(wall_length)
            
            # reading each file
            for item in os.listdir(folder_path):
                # # Summary data - Probably not needed 
                if item.endswith(".txt"):
                    summary_data = read_summary_file(folder_path + f"/{item}")
                #     wall_length = summary_data["Wall size (l)"]
                    L = summary_data["Size of box (L)"]
                    r0 = summary_data["Interaction radius (r0)"]

                # Reading orienatation data
                if item.endswith(".npz") & (item.split("_")[0] == "orientations"):
                    data = {}
                    sim_data = np.load(folder_path + "/" + item)
                    data["orientations"] = sim_data["orientations"]
                    eta = float(sim_data["noise"])
                    data["eta"] = eta
                    data["wall_length"] = wall_length

                    # Add to the big data dictionary list
                    orientation_data.append(data)

                    # Adding all the etas to be stored
                    if eta not in etas:
                        etas.append(eta)
    return orientation_data#, wall_lengths, etas

def compress_data():
    """Compresses the data from `read_orientations` function, stored in both functions as orientation_data"""

    global wall_lengths, etas
    orientation_data = read_orientations()
    compressed_data = []
    # Loop through wall_lengths and etas, if they are the same then average that data
    for wall_length in wall_lengths:
        # print(wall_length)
        for eta in etas:
            matching_orientations = []
            for data in orientation_data:
                if (data["wall_length"] == wall_length) and (data["eta"]==eta):
                    matching_orientations.append(data["orientations"])
                
            if matching_orientations:
                # Find the minimum length to ensure consistent array shapes
                min_length = min(len(orient) for orient in matching_orientations)
                
                # Truncate all arrays to the minimum length
                truncated_orientations = [orient[:min_length] for orient in matching_orientations]
                
                # Now we can safely create a NumPy array and calculate the mean
                temp_data = {
                    "wall_length": wall_length,
                    "eta": eta,
                    "orientations": np.array(truncated_orientations).mean(axis=0)
                }
                compressed_data.append(temp_data)
                
    return compressed_data
        

def read_individual(wall_length, eta, iteration,start_index):
    filepath = dir+ f"/{dir_starter}_{wall_length}_10000/orientations_{eta}_{iteration}.npz"
    sim_data = np.load(filepath)["orientations"][start_index:]
    return sim_data

def plot_noise_dependance(compressed_data, target_wall_length, steady_state_start_index=0):
    noise_orient_pairs = []

    for data in compressed_data:
        if data["wall_length"] == target_wall_length:
            noise = data["eta"]
            avg_orientation = np.mean(data["orientations"][steady_state_start_index:])
            noise_orient_pairs.append((noise, avg_orientation))
    
    # Sort by noise values
    noise_orient_pairs.sort(key=lambda x: x[0])
    noise_values, final_orientations = zip(*noise_orient_pairs)
    
    return np.array(noise_values), np.array(final_orientations)

compressed_data = compress_data()

if timeav_noise:
    fig, ax = plt.subplots(figsize = (fig_width,fig_height))
    for wall_length in wall_lengths:
        noise_values, final_orientations = plot_noise_dependance(compressed_data, target_wall_length=wall_length,steady_state_start_index=3000)
        wall_label = float(wall_length)
        ax.plot(noise_values,final_orientations, label = rf"$l$: {Fraction(wall_label/64.0).limit_denominator()}$L$", marker  = ".")

    ax.legend(frameon = False)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle \varphi \rangle_t$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(f"figures/noise_dependance.png")


# Reading an indivdual case
wall_length = wall_lengths[2]
start_index = 0
eta = 0.1
i = 2
if single_noise:
    fig,ax2 = plt.subplots()
    for i in range(3):
        sim_data = read_individual(wall_length,eta,i,start_index)
        x = np.arange(0,len(sim_data),1)+start_index
        ax2.plot(x,sim_data, label = f"{i}")
    ax2.set_title(fr"Average Orientation, $\eta$: {eta}, $l$: {Fraction(float(wall_length)/64.0).limit_denominator()}$L$")
    ax2.legend(frameon = False)
    ax2.set_ylabel(r'$\varphi$')
    ax2.set_xlabel('Time Step')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig.tight_layout()
    

def get_averaged_histograms(target_eta, target_wall_length):
    """
    Gets averaged histograms for a specific eta and wall_length
    
    Parameters:
    target_eta (float): Noise parameter to filter by
    target_wall_length (str): Wall length to filter by
    
    Returns:
    tuple: (averaged_steady_histogram, averaged_transient_histogram)
    """    
    steady_histograms = []
    transient_histograms = []
    
    # Only process folders matching the target wall length
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter and exp_item.split("_")[1] == target_wall_length:
            folder_path = os.path.join(dir, exp_item)
            
            # Only process histogram files matching the target eta
            for item in os.listdir(folder_path):
                if item.endswith(".npz"):
                    # Check if it's a histogram file with the target eta
                    parts = item.split("_")
                    
                    # Only process if it's a histogram file
                    if len(parts) >= 3 and parts[1] == "histogram":
                        file_eta = float(parts[-2])  # Eta is the second-to-last element
                        
                        # Only process if eta matches
                        if file_eta == target_eta:
                            file_path = os.path.join(folder_path, item)
                            histogram_data = np.load(file_path)["hist"]
                            iteration = parts[-1].split(".")[0]  # Get iteration number
                            
                            # Add to the appropriate list
                            if parts[0] == "steady":
                                steady_histograms.append(histogram_data)
                            elif parts[0] == "transient":
                                transient_histograms.append(histogram_data)
    
    # Ensure all histograms have the same shape
    if steady_histograms:
        min_shape = min(hist.shape for hist in steady_histograms)
        steady_histograms = [hist[:min_shape[0], :min_shape[1]] for hist in steady_histograms]
        averaged_steady = np.mean(steady_histograms, axis=0)
    else:
        averaged_steady = None

    if transient_histograms:
        min_shape = min(hist.shape for hist in transient_histograms)
        transient_histograms = [hist[:min_shape[0], :min_shape[1]] for hist in transient_histograms]
        averaged_transient = np.mean(transient_histograms, axis=0)
    else:
        averaged_transient = None
    
    return averaged_steady, averaged_transient

def get_all_histogram_params():
    """
    Returns all available eta and wall_length combinations in the dataset
    
    Returns:
    tuple: (available_etas, available_wall_lengths)
    """
    
    available_etas = set()
    available_wall_lengths = set()
    
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter:
            wall_length = exp_item.split("_")[1]
            available_wall_lengths.add(wall_length)
            
            folder_path = os.path.join(dir, exp_item)
            for item in os.listdir(folder_path):
                if item.endswith(".npz") and "_histogram_" in item:
                    parts = item.split("_")
                    if len(parts) >= 3:
                        try:
                            eta = float(parts[-2])
                            available_etas.add(eta)
                        except ValueError:
                            continue
    
    return sorted(list(available_etas)), sorted(list(available_wall_lengths))

wall_length_ind = 1
eta_ind = 2

hist_type = ["steady", "transient"][0]
etas, wall_lengths = get_all_histogram_params()
wall_length = wall_lengths[wall_length_ind]
eta = etas[eta_ind]

histograms = False
if histograms:
    # etas, wall_lengths = get_all_histogram_params()
    steady, transient  = get_averaged_histograms(eta, wall_length)
    if hist_type == "steady":
        hist = steady
    elif hist_type == "transient":
        hist = transient
    else: 
        raise ValueError("hist_type must be either 'steady' or 'transient'")

    fig3, ax3 = plt.subplots(figsize = (fig_width,fig_height))
    cax = ax3.imshow(hist.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto') # This summs all simulaitons to find an average
    cbar = fig3.colorbar(cax, ax=ax3, ticks=[])  # Removes ticks and labels in one line
    cbar.set_label('Density')
    cbar.ax.tick_params(labelsize=0)   # Remove colorbar tick labels
    lower =  0#L*1/4
    upper = L#*3/4
    ax3.set_title("Density Histogram")
    ax3.set_xlim(lower, upper)
    ax3.set_ylim(lower, upper)
    ax3.set_aspect('equal')  # Ensure square aspect ratio
    ax3.set_xticks(np.linspace(lower,upper,5))
    ax3.set_yticks(np.linspace(lower,upper,5))
    ax3.set_aspect("equal")
    fig3.tight_layout()


def find_near_region(histogram, threshold_percentage=0.5, mult_r0=3):
    av_density = np.mean(histogram)
    threshold_density = av_density * threshold_percentage
    low_density_mask = histogram < threshold_density

    masked_hist = histogram.copy()
    # masked_hist = np.ones_like(histogram)
    wall_x = L / 2

    n_cells = len(masked_hist)
    cell_width = L / n_cells
    for y in range(masked_hist.shape[1]):
        for x in range(masked_hist.shape[0]):
            dist_to_wall = abs((x + 0.5) * cell_width - wall_x)
            if low_density_mask[x, y] and dist_to_wall <= mult_r0 * r0:
                masked_hist[x, y] = 0

    return masked_hist

def average_distance_from_wall(masked_hist, wall_length):
    wall_x = L/2
    max_y = L/2 + wall_length*0.5
    min_y = L/2 - wall_length*0.5
    distances = []
    n_cells = len(masked_hist)
    cell_width = L/n_cells
    # Loop through all the y
    for y in range(masked_hist.shape[1]):
        # Check if in the bounds of the wall
        if min_y <= y*L/n_cells <=max_y:
            # Loop through x until the first 0
            for x in range(masked_hist.shape[0]): # approaching the wall from the left
                if masked_hist[x, y] == 0:
                    
                    dist_to_wall = abs((x + 0.5) * cell_width - wall_x)
                    
                    # print("left",x*L/n_cells,y*L/n_cells, dist_to_wall)
                    distances.append(dist_to_wall)
                    break  # Stop after finding the first 0 in the x direction
            for x in range(masked_hist.shape[0]-1, 0,-1): # approaching the wall from the right
                if masked_hist[x, y] == 0:
                    
                    dist_to_wall = abs((x + 0.5) * cell_width - wall_x)                   
                    # print("right", x*L/n_cells,y*L/n_cells, dist_to_wall)
                    distances.append(dist_to_wall)
                    break 

    if distances:
        return np.mean(distances) +cell_width/2, np.std(distances)  # Average the distances + distance to the edge of the cell, standard dev
    else:
        return None

bin_area = (L/320)**2 #check this value!
if histogram_filters:
    threshold = 0.3
    max_distance = 2
    wall_length = wall_lengths[wall_length_ind]
    eta = 0.4# etas[eta_ind]
    histogram,_ = get_averaged_histograms(eta, wall_length)
    # print(average_distance_from_wall(find_near_region(histogram,threshold_percentage=threshold, mult_r0=max_distance), float(wall_length)))
    # print(histogram)
    # print(find_near_region(histogram))

    # Displaying the histogram with and without the filter
    fig4, ax4 = plt.subplots(ncols = 2, figsize = (fig_width*2,fig_height))
    cax = ax4[0].imshow(find_near_region(histogram, threshold, mult_r0=max_distance).T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    cax2 = ax4[1].imshow(histogram.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    cbar = fig4.colorbar(cax, ax=ax4)

    ## Finding the distance based on the filter
    fig5, ax5 = plt.subplots(figsize = (fig_width,fig_height))

    for wall_length in wall_lengths:
        distances = []
        deviations = []
        for eta in etas:
            steady_hist,_ = get_averaged_histograms(eta, wall_length)
            mean, std = average_distance_from_wall(find_near_region(histogram,threshold_percentage=threshold), float(wall_length))
            distances.append(mean)
            deviations.append(std)

        ax5.plot(etas, distances, label = f"l={Fraction(float(wall_length)/L).limit_denominator(3)}L")
    ax5.set_xlabel(r"$\eta$")
    ax5.set_ylabel(r"D ($R_0$)")
    ax5.legend(frameon=False)
    fig5.tight_layout()

    vmin =0
    vmax = 2
    fig6, ax6 = plt.subplots(nrows=2, ncols=1, figsize= (fig_width, fig_height*2))
    I1 = get_averaged_histograms(etas[1], wall_lengths[-3])[0].T/bin_area
    cax4 = ax6[0].imshow(I1, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto',vmin=vmin, vmax=vmax)
    ax6[0].set_xticks([])  # Remove x-axis ticks for the first subplot
    # ax6[0].set_xlabel('')  # Optionally, remove x-axis label for the first subplot
    I2 = get_averaged_histograms(etas[-1], wall_lengths[-3])[0].T/bin_area
    cax5 = ax6[1].imshow(I2, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto',vmin=vmin, vmax=vmax)
    fig6.tight_layout()


def count_iterations(target_eta, target_wall_length):
    """Count how many iterations were averaged for a specific eta and wall_length"""
    count = 0
    folder_path = dir + f"/{dir_starter}_{target_wall_length}_10000"
    
    for item in os.listdir(folder_path):
        if item.startswith("steady_histogram_data") and item.endswith(".npz"):
            parts = item.split("_")
            if len(parts) >= 4:
                try:
                    eta = float(parts[-2])
                    if eta == target_eta:
                        count += 1
                except ValueError:
                    continue
    
    return max(1, count)

if Density_profile:
    fig7, ax7 = plt.subplots(figsize = (fig_width,fig_height))

    wall_length = wall_lengths[-1]
    sim_params = read_summary_file(dir + f"/{dir_starter}_{wall_length}_10000/simulation_parameters_{wall_length}.txt")
    wall_label = float(wall_length)
    nsteps = int(sim_params["Total number of steps"])
    N = int(sim_params["Number of particles (N)"])
    steady_state_steps = 5e3


    for eta in [0.2,0.3, 0.4,0.6]:
        histogram = get_averaged_histograms(eta, wall_length)[0].T
        # Proper normalization:
            # 1. Divide by bin_area to get density per area
            # 2. Divide by number of particles to normalize by particle count
            # 3. Divide by number of time steps to get average over time
            # 4. Divide by number of iterations that were averaged
        iterations_count = count_iterations(eta, wall_length)
        print(iterations_count)

        I1 = histogram/(bin_area*N*steady_state_steps* iterations_count)
        # I1 /=I1.mean()
        x = np.linspace(0,L, I1.shape[0])
        ax7.plot(x,I1.mean(axis=0),label = r"$\eta$: " +f"{eta}")

    ax7.legend(frameon = False, loc = "lower right")
    ax7.set_xticks(np.linspace(0, L, 5))
    ax7.set_xlim(0,L)
    ax7.set_ylabel(r"Density ($R_0^{-2}$)")
    ax7.set_xlabel(r"x ($R_0$)")
    # ax7.set_title(f"Density Profile" + rf" $l$: {Fraction(wall_label/L).limit_denominator()}$L$")
    # ax7.grid()
    # ax7.axhline(y=2.5e-5, linestyle="--", color="grey", alpha = 0.5)
    fig7.tight_layout()
    fig7.savefig(f"figures/density_profile_Etas_{wall_label/L:.2f}.png")

if Density_profile2:
    fig8, ax8 = plt.subplots(figsize = (fig_width,fig_height))

    wall_length = wall_lengths[-3]
    sim_params = read_summary_file(dir + f"/{dir_starter}_{wall_length}_10000/simulation_parameters_{wall_length}.txt")
    wall_label = float(wall_length)
    nsteps = int(sim_params["Total number of steps"])
    N = int(sim_params["Number of particles (N)"])
    steady_state_steps = 5e3

    eta = etas[1]
    
    for wall_length_str in wall_lengths:
        histogram = get_averaged_histograms(eta, wall_length_str)[0].T
        # Proper normalization:
            # 1. Divide by bin_area to get density per area
            # 2. Divide by number of particles to normalize by particle count
            # 3. Divide by number of time steps to get average over time
            # 4. Divide by number of iterations that were averaged
        iterations_count = count_iterations(eta, wall_length_str)
        print(iterations_count)

        I1 = histogram#/(bin_area*N*steady_state_steps* iterations_count)
        I1 /=I1.sum()
        
        x = np.linspace(0,L, I1.shape[0])
        wall_length_float = float(wall_length_str)
        ax8.plot(x,I1.mean(axis=0),label = r"$l$: " +f"${Fraction(wall_length_float/L).limit_denominator()}$L")

    ax8.legend(frameon = False, loc = "lower right")
    ax8.set_xticks(np.linspace(0, L, 5))
    ax8.set_xlim(0,L)
    ax8.set_ylabel(r"Density ($R_0^{-2}$)")
    ax8.set_xlabel(r"x ($R_0$)")
    # ax8.set_title(f"Density Profile" + rf" $\eta$: {eta}")
    # ax8.grid()
    # ax8.axhline(y=2.5e-5, linestyle="--", color="grey", alpha = 0.5)
    # fig8.savefig(f"figures/density_profile_wallLengths_{eta}.png")


    # max_density = []
    # wall_floats = []
    # fig9,ax9 = plt.subplots(figsize = (fig_width,fig_height))
    # wall_length_str = wall_lengths[-1]
    # for eta in etas:
    #     eta_float = float(eta)

    #     histogram = get_averaged_histograms(eta, wall_length_str)[0].T

    #     I1 = histogram#/(bin_area*N*steady_state_steps* iterations_count)
    #     I1 /=I1.sum()

    #     max_density.append(I1.max())
    #     wall_floats.append(eta)

    # ax9.plot(wall_floats,max_density, marker = "x")
    # ax9.set_xlabel(r"$\eta$")
    # ax9.set_ylabel(r"Max Density")
    # ax9.set_ylim(0,0.0001)


plt.show()