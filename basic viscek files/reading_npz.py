import numpy as np
import os
import pandas as pd
dir = "npzfiles"
len_dir = len(os.listdir(dir))

loaded_positions = []
loaded_angles = [] # np.empty(len_dir)

for i,filename in enumerate(os.listdir(dir)):
    filepath = os.path.join(dir, filename)

    data = np.load(filepath)   
    loaded_positions.append( data["positions"])
    loaded_angles.append( data["angles"])
loaded_positions = np.array(loaded_positions)
loaded_angles = np.array(loaded_angles)


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

# Display the first few rows of the DataFrame
print(df)