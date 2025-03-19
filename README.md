# WettingRobots
Directory for Rory Tidmarsh's final year project on wetting robots

## Description 

├── `basic viscek files` folder: the original code given by F.Turci, our break down of the code using 'for' loops and reading the .npz file.

├──**`wall implementation`** folder: folder containing implemetation of the wall

│    ├──`wall_size_experiment` (dir): folder containing the experimental data for varying the wall

│    │    ├──`test128_{l}_{nsteps}`... (dir): folders containtin .npz files for wall length `l`

│    │    ├──`analysis.py`: python file containing the analysis figure creation for the .npz files  

│    │    ├──`finalstate.npz`: final state of the result from the `Vicsek wall animation`

│   ├──`arc_analysis` (dir): folder containing the experimental data for varying arc subtended angle and confined geometry for a full circle (`.npz` files). Plus the analysis file

│   ├──`noise experiment` (dir): experiment for varying noise to get order parameter graph with analysis files inside

│   ├──`Vicsek_wall_animation`: python file containing animation of the system

│   ├──`Vicsek_wall.py`: python file containing ability to output data to `.npz` files for later analysis, all saved into the `wall_size_experiment` directory unless noise is varied where it is saved into the `noise_experiment` directionry 

│   ├──`Vicsek_arc.py`: equvialnt to wall but with a curved geometry

│   ├──`Vicsek_arc_animation.py`: 
        
├──`mygradient` (dir): start of kilombo code

├──`mini-quarto-website` (dir): directory for the website that runs the animations.


## Packages
- `numpy` for numerical computations
- `matplotlib` for plotting
- `numba` for speed up
- `pycicrclar` for angular statistics, this requires the `statsmodels`
- `tqdm` for progress bars
