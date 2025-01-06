# WettingRobots

## Description 
"basic viscek files" folder: the original code given by F.Turci, our break down of the code using 'for' loops and reading the .npz file.

"wall implementation" folder looks at implementing a wall into the viscek model. It contains:
- "Viscek_wall": **MAIN WALL IMPLEMENTATION**. The wall implemented using numba to speed up and grid searching to find nearest neighbours
- "wall_size_experiment/": folder containing the data for comparing how how the size of the wall affects the system. Data is left out from gitignore file but the jupyternotebook is there for the reading of the data
