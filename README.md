# WettingRobots
Directory for Rory Tidmarsh's final year project on wetting robots
## Description 
|-- `basic viscek files` folder: the original code given by F.Turci, our break down of the code using 'for' loops and reading the .npz file.

|--**`wall implementation`** folder: folder containing implemetation of the wall **Interim report based on this folder**

    |---`Vicsek_wall_animation`: python file containing animation of the system
    
    |---`Vicsek_wall.py`: python file containing ability to output data to `.npz` files for later analysis, all saved into the `wall_size_experiment` directory
    
    |---`wall_size_experiment` (dir): folder containing the experimental data for varying the wall
    
        |---`test128_{l}_{nsteps}`... (dir): folders containtin .npz files for wall length `l`
        
        |---`analysis.py`: python file containing the analysis figure creation for the .npz files
        
        |---`finalstate.npz`: final state of the result from the `Vicsek wall animation`
        
 |--`mygradient` (dir): start of kilombo code


## Packages
- `numpy` for numerical computations
- `matplotlib` for plotting
- `numba` for speed up
- `pycicrclar` for angular statistics
