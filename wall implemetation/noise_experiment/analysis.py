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

text_width = 10
fig_width = text_width
fig_height = 0.8* fig_width 

width_2_subplot = fig_width/2 + 1
height_2_subplot = 0.75*width_2_subplot
height_cbar_2_subplot = 0.75*width_2_subplot

histograms = True
timeav_noise = False
single_noise = False

scale = 1
plt.rcParams.update({
    'font.size': 28*scale,
    'axes.labelsize': 28*scale,
    'axes.titlesize': 36*scale,
    'xtick.labelsize': 28*scale,
    'ytick.labelsize': 28*scale,
    'legend.fontsize': 27*scale
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
    global wall_lengths, etas,L
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
                # temp_data = {"wall_length": wall_length, "eta": eta}
                # temp_orientations = []
                if (data["wall_length"] == wall_length) and (data["eta"]==eta):
                    matching_orientations.append( data["orientations"])
                
            if matching_orientations:
                # print("Found matching orientations for wall length: ", wall_length, " and eta: ", eta)
                temp_data = {
                    "wall_length": wall_length,
                    "eta": eta,
                    "orientations": np.array(matching_orientations).mean(axis=0)
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
    import os
    import numpy as np
    
    # Set directory
    dir = str(os.getcwd()) + "/wall implemetation/noise_experiment"
    dir_starter = "DistanceNoise64"
    
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
    
    # Average the histograms if any were found
    averaged_steady = np.mean(steady_histograms, axis=0) if steady_histograms else None
    averaged_transient = np.mean(transient_histograms, axis=0) if transient_histograms else None
    
    return averaged_steady, averaged_transient

def get_all_histogram_params():
    """
    Returns all available eta and wall_length combinations in the dataset
    
    Returns:
    tuple: (available_etas, available_wall_lengths)
    """
    import os
    
    dir = str(os.getcwd()) + "/wall implemetation/noise_experiment"
    dir_starter = "DistanceNoise64"
    
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

wall_length_ind = -1
eta_ind = 2
# L = 64
hist_type = ["steady", "transient"][0]
if histograms:
    etas, wall_lengths = get_all_histogram_params()
    wall_length = wall_lengths[wall_length_ind]
    # eta = etas[eta_ind]
    eta = 0.5

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

plt.show()