import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "/wall implemetation/wall_size_experiment//128wall"
filenames = os.listdir(dir)
eta = 0.1
# dir_starter = ["0.3noise50","wall50",][0]
dir_starter = f"{eta}noise128"
save_dir = str(os.getcwd())+ "\wall implemetation//wall_size_experiment//128wall//figures"
# print(filenames)
cmap = "hsv"
L = 128
# file_starter = "wall"

i = 1

text_width = 10
fig_width = text_width
fig_height = 0.75* fig_width 

width_2_subplot = fig_width/2 + 1
height_2_subplot = 0.75*width_2_subplot
height_cbar_2_subplot = 0.75*width_2_subplot

scale = 1
plt.rcParams.update({
    'font.size': 28*scale,
    'axes.labelsize': 28*scale,
    'axes.titlesize': 36*scale,
    'xtick.labelsize': 28*scale,
    'ytick.labelsize': 28*scale,
    'legend.fontsize': 28*scale
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
    global wall_lengths, etas
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
                # if item.endswith(".txt"):
                #     summary_data = read_summary_file(folder_path + f"/{item}")
                #     wall_length = summary_data["Wall size (l)"]

                # Reading orienatation data
                if item.endswith(".npz"):
                    if item.split("_")[0] == "orientations":
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

# fig, ax = plt.subplots(figsize = (fig_width,fig_height))
# for wall_length in wall_lengths:
#     noise_values, final_orientations = plot_noise_dependance(compressed_data, target_wall_length=wall_length,steady_state_start_index=3000)
#     wall_label = float(wall_length)
#     ax.plot(noise_values,final_orientations, label = rf"$l$: {Fraction(wall_label/64.0).limit_denominator(3)}$L$", marker  = ".")

# ax.legend(frameon = False)
# ax.set_xlabel(r"$\eta$")
# ax.set_ylabel(r"$\langle \varphi \rangle_t$")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()


# Reading an indivdual case
wall_length = wall_lengths[-1]
start_index = 3000
# eta = 0.3
i = 1
fig,ax2 = plt.subplots(figsize = (fig_width,fig_height))
for i in range(3):
    sim_data = read_individual(wall_length,eta,i,start_index)
    x = np.arange(0,len(sim_data),1)+start_index
    ax2.plot(x,sim_data, label = f"{i}")
# ax2.set_title(fr"Average Orientation, $\eta$: {eta}, $l$: {Fraction(float(wall_length)/L).limit_denominator(3)}$L$")
ax2.legend(frameon = False)
ax2.set_ylabel(r'$\varphi$')
ax2.set_xlabel('Time Step')
ax2.set_ylim(0,1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig.tight_layout()
plt.savefig(f"{save_dir}\\50wall_eta{eta}.png", dpi = 300)
plt.show()