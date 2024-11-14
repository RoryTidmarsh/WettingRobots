import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
dir = "npzfiles"
filenames = os.listdir(dir)
len_dir = len(filenames)

filenames = filenames[1:]

loaded_positions = []
loaded_angles = [] # np.empty(len_dir)

for filename in filenames:
    filepath = os.path.join(dir, filename)

    data = np.load(filepath)   
    loaded_positions.append( data["positions"])
    loaded_angles.append( data["angles"])
loaded_positions = np.array(loaded_positions)
loaded_angles = np.array(loaded_angles)

def read_summary_file(filepath):
    summary_data = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Split the line into key and value
            key, value = line.strip().split(': ')
            # Store the key-value pair in the dictionary
            summary_data[key] = float(value) if '.' in value or value.isdigit() else value
    return summary_data

dir = "npzfiles"
summary_file = os.listdir(dir)[0]
summary_filepath = os.path.join(dir, summary_file)
summary_values = read_summary_file(summary_filepath)
print("Summary Values:", summary_values)

def IntoDataframe(loaded_positions, loaded_angles):
    ## From here it just puts it into a dataframe
    nfiles = loaded_positions.shape[0]
    nparticles = loaded_positions.shape[1]

    # Create a MultiIndex for the DataFrame
    index = pd.MultiIndex.from_product([range(nfiles), range(nparticles)], names=["File", "Particle"])

    # Reshape the loaded positions and angles for the DataFrame
    positions_flat = loaded_positions.reshape(-1, 2)  # Shape will be (nfiles * nparticles, 2)
    angles_flat = loaded_angles.flatten()              # Shape will be (nfiles * nparticles,)

    # Create the DataFrame
    df = pd.DataFrame({
        "Positions": list(positions_flat),  # List of arrays for positions
        "Angles": angles_flat                # Flattened angles
    }, index=index)
    return df

def histogram2D(loaded_positions, plot= None):
    L, r0 = summary_values["Size of box (L)"], summary_values["Interaction radius (r0)"]
    positions = loaded_positions[0]
    bins = int(L/(r0/2))

    hist, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= bins, density = False)
    for i in range (1, int(summary_values["Total number of steps"])):
        positions = loaded_positions[i]
        hist_new,_,_= np.histogram2d(positions[:, 0], positions[:,1], bins= bins, density = False)
        hist += hist_new

    if plot ==None:
        fig, ax =plt.subplots(figsize = (6,6))
    else: 
        fig, ax = plot
    hist_normalised = hist.T/sum(hist)
    
    # Use imshow to display the normalized histogram
    cax = ax.imshow(hist_normalised, extent=[0, L, 0, L], origin='lower', cmap='hot', aspect='auto')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Normalised 2D Histogram of Particle Positions")
    # Add a colorbar for reference
    fig.colorbar(cax, ax=ax, label='Density')
    return ax

fig,ax = plt.subplots(figsize = (12,6), ncols = 2)
ax[0] = histogram2D(loaded_positions=loaded_positions, plot= (fig,ax[0]))
plt.show()