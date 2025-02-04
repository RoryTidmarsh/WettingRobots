import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "/wall implemetation/noise_experiment"
filenames = os.listdir(dir)
dir_starter = "noise64"
# print(filenames)
cmap = "hsv"
L = 128
file_starter = "test128"

stream_plots = False
alignment = True
histograms = False
final_positions = False


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
        

def read_individual(wall_length, eta, iteration):
    folder_path = dir + "/" + f"{file_starter}_{wall_length}_10000"
    filepath = folder_path + "/" + f"orientations_{eta}_{iteration}.npz"
    sim_data = np.load(filepath)

    return sim_data["orientations"]

def plot_noise_dependance(compressed_data, target_wall_length, steady_state_start_index = 0):
    noise_values = []
    final_orientations = []

    for data in compressed_data:
        if data["wall_length"] == target_wall_length:
            # print(data)
            noise_values.append(data["eta"])
            # final_orientations.append(data["orientations"][steady_state_start_index:])
            average_orientation = np.mean(data["orientations"][steady_state_start_index:])
            final_orientations.append(average_orientation)
    assert len(noise_values) == len(final_orientations)
    return noise_values, final_orientations

compressed_data = compress_data()

fig, ax = plt.subplots()
for wall_length in wall_lengths:
    noise_values, final_orientations = plot_noise_dependance(compressed_data, target_wall_length=wall_length,steady_state_start_index=3000)
    wall_label = float(wall_length)
    ax.plot(noise_values,final_orientations, label = rf"$l$: {Fraction(wall_label/64.0).limit_denominator()}$L$")

ax.legend(frameon = False)
ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$\langle \varphi \rangle_t$")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
