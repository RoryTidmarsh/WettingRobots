import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "/wall implemetation/wall_size_experiment/64L"
filenames = os.listdir(dir)
dir_starter = "wall64"
print(filenames)

# dir2 = str(os.getcwd())+ "/wall implemetation/wall_size_experiment/64L/wall64_0.0_10000"
# filenames2 = os.listdir(dir2)
# print(filenames2)
cmap = "hsv"
L = 64
r0 = 1

single_hist = False
average_hist = False
stream = False  # Useless if direction_histogram on
quiver = False  # Useless if direction_histogram on
alignment_direction = False
alignemnt_spread = False
direction_histogram = True

dual_hitogram = False # putting two histograms in one plot

save = False
save_dir = str(os.getcwd())+"/wall implemetation/wall_size_experiment\\figures\\64L"

def get_params():
    available_walls = set() # multiples of pi
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter:
            wall = exp_item.split("_")[1]
            available_walls.add(wall)
    if not available_walls:
        raise ValueError(f"No available walls found. The directory starter '{dir_starter}' might be incorrect.")
    return sorted(list(available_walls))
available_walls = get_params()
wall_length = available_walls[-2]
wall_length_float = float(wall_length)
iteration = 4
wall_yMin = L/2 - wall_length_float/2
wall_yMax = L/2 + wall_length_float/2

## Figure information
text_width = 3.25  # inches (single column width)
fig_width = text_width
fig_height = 0.8 * fig_width  # for standard plots
width_2_subplot = fig_width/2 + 0.25  # for side-by-side subplots
height_2_subplot = 0.75 * width_2_subplot
height_cbar_2_subplot = 0.75 * width_2_subplot
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
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
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.97,
    'figure.subplot.bottom': 0.12,
    'figure.subplot.top': 0.92,
    'figure.dpi': 300,
    'savefig.dpi': 300,
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

filepath = dir + "\\wall64_0.0_10000\\simulation_parameters_0.0.txt"

summary_data = read_summary_file(filepath)

L = int(summary_data['Size of box (L)'])
N = int(summary_data['Number of particles (N)'])
eta = summary_data['Noise/randomness (eta)']
r0 = float(summary_data['Interaction radius (r0)'])
rho = float(summary_data['Density (rho)'])
eta = float(summary_data['Noise/randomness (eta)'])
wall_x = L/2
averaged_frames = float(summary_data['Alignment average steps'])

def count_iterations(target_wall_length):
    """Count how many iterations were averaged for a specific wall_length"""
    count = 0
    folder_path = dir + f"/{dir_starter}_{target_wall_length}_10000"
    
    for item in os.listdir(folder_path):
        if item.startswith("steady_histogram_data") and item.endswith(".npz"):
            parts = item.split("_")
            if len(parts) >= 4:
                try:
                    wall_length = float(parts[-2])
                    if wall_length == float(target_wall_length):
                        count += 1
                except ValueError:
                    continue
    
    return max(1, count)

def get_histogram(wall_length, iteration):
    steady_state_steps = 7e5
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter and exp_item.split("_")[1] == wall_length:
            folder_path = os.path.join(dir, exp_item)
            for item in os.listdir(folder_path):
                if item.endswith(f"{iteration}.npz") and "steady_histogram" in item:
                    histogram_path = os.path.join(folder_path, item)
                    histogram_data = np.load(histogram_path)["hist"]

                    #normalise the histogram
                    bin_area = (L/len(histogram_data))**2
                    histogram_data = histogram_data/(np.sum(histogram_data))

                    return histogram_data

def get_averaged_histograms(target_wall_length):
    """
    Gets averaged histograms for a specific wall_length
    
    Parameters:
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
            
            # Only process histogram files
            for item in os.listdir(folder_path):
                if item.endswith(".npz"):
                    # Check if it's a histogram file
                    parts = item.split("_")
                    
                    # Only process if it's a histogram file
                    if len(parts) >= 3 and parts[1] == "histogram":
                        file_path = os.path.join(folder_path, item)
                        histogram_data = np.load(file_path)["hist"]
                        iteration = parts[-1].split(".")[0]  # Get iteration number
                        
                        # Normalize the histogram
                        bin_area = (L / len(histogram_data)) ** 2
                        histogram_data = histogram_data / (bin_area * N * averaged_frames)
                        
                        # Add to the appropriate list
                        if parts[0] == "steady":
                            steady_histograms.append(histogram_data)
                        elif parts[0] == "transient":
                            transient_histograms.append(histogram_data)
    
    # Average the histograms
    if steady_histograms:
        min_shape = min(hist.shape for hist in steady_histograms)
        steady_histograms = [hist[:min_shape[0], :min_shape[1]] for hist in steady_histograms]
        averaged_steady = np.mean(steady_histograms, axis=0)
        averaged_steady = averaged_steady / np.sum(averaged_steady)  # Normalize to avoid overflow
    else:
        averaged_steady = None

    if transient_histograms:
        min_shape = min(hist.shape for hist in transient_histograms)
        transient_histograms = [hist[:min_shape[0], :min_shape[1]] for hist in transient_histograms]
        averaged_transient = np.mean(transient_histograms, axis=0)
        averaged_transient = averaged_transient / np.sum(averaged_transient)  # Normalize to avoid overflow
    else:
        averaged_transient = None
    
    return averaged_steady, averaged_transient

def plot_x_wall(ax,wall_yMin,wall_yMax, wall_color = "blue", boundary = True, walpha = 1, *args, **kwargs):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
    wall_distance = r0
    if boundary==True:
        # Boundary left and right of the wall within vertical bounds
        ax.plot([wall_x - wall_distance, wall_x - wall_distance], [wall_yMin, wall_yMax], 'b--', lw=2, label=f'Boundary at {wall_distance:.2f}')
        ax.plot([wall_x + wall_distance, wall_x + wall_distance], [wall_yMin, wall_yMax], 'b--', lw=2)
        
        # Boundary above the wall (top circle segment)
        theta = np.linspace(0, np.pi, 100)  # For the top part of the wall
        top_circle_x = wall_x + wall_distance * np.cos(theta)
        top_circle_y = wall_yMax + wall_distance * np.sin(theta)
        ax.plot(top_circle_x, top_circle_y, 'b--', lw=2)
        
        # Boundary below the wall (bottom circle segment)
        theta = np.linspace(np.pi, 2 * np.pi, 100)  # For the bottom part of the wall
        bottom_circle_x = wall_x + wall_distance * np.cos(theta)
        bottom_circle_y = wall_yMin + wall_distance * np.sin(theta)
        ax.plot(bottom_circle_x, bottom_circle_y, 'b--', lw=2)

    #plot the wall
    ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall", color = wall_color, alpha = walpha, *args, **kwargs)
    return ax

if single_hist:

    histogram_data = get_histogram(wall_length, iteration)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height),constrained_layout=True)
    cax = ax.imshow(histogram_data.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    if wall_length_float != 0:
        ax = plot_x_wall(ax,wall_yMin=wall_yMin,wall_yMax=wall_yMax, boundary=False,wall_color="r")
    
    ax.legend()
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Density")
    ax.set_xlabel(r"x ($R_0$)")
    ax.set_ylabel("y ($R_0$)")
    ax.set_aspect("equal")
    
    upper, lower = L/2+1.1*wall_length_float/2, L/2-1.1*wall_length_float/2
    ax.set_xlim(upper, lower)
    ax.set_ylim(upper, lower)
    

    #save fig
    if save:
        fig.savefig(f"{save_dir}/wall64_{wall_length_float:.2f}_histogramZOOMED_{iteration}.png")

if average_hist:
    histogram_data = get_averaged_histograms(wall_length)[0]
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height),constrained_layout=True)
    cax1 = ax1.imshow(histogram_data.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')

    if wall_length_float != 0:
        ax1 = plot_x_wall(ax1,wall_yMin=wall_yMin,wall_yMax=wall_yMax, boundary=False,wall_color="r")
    
    ax1.legend()
    cbar1 = fig1.colorbar(cax1, ax=ax1)
    cbar1.set_label("Density")
    ax1.set_xlabel(r"x ($R_0$)")
    ax1.set_ylabel("y ($R_0$)")
    ax1.set_aspect("equal")

    #save fig
    if save:
        fig1.savefig(f"{save_dir}/wall64_{wall_length}_histogram_avg.png")

def get_average_orientation(wall_length, iteration):
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter and exp_item.split("_")[1] == wall_length:
            folder_path = os.path.join(dir, exp_item)
            for item in os.listdir(folder_path):
                if item.endswith(f"{iteration}.npz") and "alignment" in item:
                    orientation_path = os.path.join(folder_path, item)
                    orientation_data = np.load(orientation_path)["angles"]
                    return orientation_data
# print(len(get_average_orientation(wall_length, iteration)))

def get_orientation_spread(wall_length, iteration):
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter and exp_item.split("_")[1] == wall_length:
            folder_path = os.path.join(dir, exp_item)
            for item in os.listdir(folder_path):
                if item.endswith(f"{iteration}.npz") and "orientations" in item:
                    spread_path = os.path.join(folder_path, item)
                    spread_data = np.load(spread_path)["orientations"]
                    return spread_data
# print(len(get_orientation_spread(wall_length, iteration)))

def average_spread(wall_length):
    tot_iterations = count_iterations(wall_length)
    spread = np.array(get_orientation_spread(wall_length, 0))
    for i in range(1,tot_iterations-1):
        spread += np.array(get_orientation_spread(wall_length, i))
    return spread/tot_iterations

if alignemnt_spread:
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    for wall_length in available_walls:
        spread = average_spread(wall_length)
        ax2.plot(spread, label = r"$l=$"+f"{Fraction(float(wall_length)/L).limit_denominator()}" + r"$L$")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel(r"$\varphi (t)$") 
    ax2.legend(frameon = False)

    #save fig
    if save:
        fig2.savefig(f"{save_dir}/wall64_alignment_spread.png")

if alignment_direction:
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
    ax3.axhline(y=-np.pi/2, color='grey',alpha = 0.7, linestyle='-.', lw=0.5)
    ax3.axhline(y=np.pi/2, color='grey',alpha = 0.7, linestyle='-.', lw=0.5)
    for WALL_LENGTH in available_walls:
        # if wall_length == "64.0":     ## To demonstrate the anti-parallel case
        #     orientation = get_average_orientation(wall_length, 5)
        # else:
        orientation = get_average_orientation(WALL_LENGTH, iteration)
        x = np.arange(len(orientation))*averaged_frames
        ax3.plot(x,orientation, label = r"$l=$"+f"{Fraction(float(WALL_LENGTH)/L).limit_denominator()}" + r"$L$")
    ax3.set_yticks(np.arange(-np.pi, np.pi + np.pi/2, np.pi/2))
    ax3.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    
    ax3.set_xlabel("Time step")
    ax3.set_ylabel(r"$\langle \theta (t) \rangle$") 
    ax3.legend(frameon = True, loc = "lower right")

    #save fig
    if save:
        # fig3.savefig(f"{save_dir}/wall64_alignment_direction_{iteration}.png")
        fig3.savefig(f"{save_dir}/wall64_alignment_direction.png")

def flow_plot(angle, iteration,ax = None, type = "stream", phase = "steady", density = 1.0, *args, **kwargs):
    if type not in ["stream","quiver"]:
        raise ValueError("type must be either 'stream' or 'quiver'")
    valid_phases = ["transient", "steady"]
    if phase not in valid_phases:
        raise ValueError(f"Invalid phase: '{phase}'. Valid options are {valid_phases}.")

    valid_phases = ["transient", "steady"]
    if phase not in valid_phases:
        raise ValueError(f"Invalid phase: '{phase}'. Valid options are {valid_phases}.")

    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter and exp_item.split("_")[1] == angle:
            folder_path = os.path.join(dir, exp_item)
            for item in os.listdir(folder_path):
                if item.endswith(f"{iteration}.npz") and f"{phase}_stream" in item:
                    simulation_path = os.path.join(folder_path, item)
                    simulation = np.load(simulation_path)
    
                    
    X,Y = simulation["X"], simulation["Y"]
    _Hx_stream, _Hy_stream = simulation["X_hist"], simulation["Y_hist"]
    try:
        counts = simulation["counts"]
    except:
        counts = np.ones_like(_Hx_stream)
    #Average the velocity
    avg_vx = np.zeros_like(X)
    avg_vy = np.zeros_like(Y)
    avg_vx[counts > 0] = _Hx_stream[counts > 0] / counts[counts > 0]
    avg_vy[counts > 0] = _Hy_stream[counts > 0] / counts[counts > 0]
    
    #Only plot inside the box, (not into and out of the walls)
    avg_vx_inner = avg_vx[1:-1,1:-1]
    avg_vy_inner = avg_vy[1:-1,1:-1]
    X_inner = X[1:-1,1:-1]*L
    Y_inner = Y[1:-1,1:-1]*L

    if ax==None:
        fig,ax = plt.subplots(1,1, figsize=(fig_width, fig_height),constrained_layout=True)

    if type == "stream":
        ax.streamplot(X_inner,Y_inner,avg_vx_inner.T,avg_vy_inner.T, density=density, *args, **kwargs)
    if type == "quiver":
        ax.quiver(X_inner,Y_inner,avg_vx_inner.T,avg_vy_inner.T, *args, **kwargs)
    return ax

if stream:
    fig4, ax4 = plt.subplots(1,1, figsize=(fig_width, fig_height),constrained_layout=True)
    ax4 = flow_plot(wall_length, iteration, ax = ax4, type = "stream", phase = "steady", density = 1.0)
    if wall_length_float != 0:
        ax4 = plot_x_wall(ax4,wall_yMin=wall_yMin,wall_yMax=wall_yMax, boundary=False,wall_color="r")
    ax4.set_xlabel(r"x ($R_0$)")
    ax4.set_ylabel(r"y ($R_0$)")
    ax4.set_aspect("equal")

    #save fig
    if save:
        fig4.savefig(f"{save_dir}/wall64_{wall_length_float:.2f}_stream_{iteration}.png")

if quiver:
    fig5, ax5 = plt.subplots(1,1, figsize=(fig_width, fig_height),constrained_layout=True)
    ax5 = flow_plot(wall_length, iteration, ax = ax5, type = "quiver", phase = "steady", scale = 5)
    if wall_length_float != 0:
        ax5 = plot_x_wall(ax5,wall_yMin=wall_yMin,wall_yMax=wall_yMax, boundary=False,wall_color="r")
    ax5.set_xlabel(r"x ($R_0$)")
    ax5.set_ylabel(r"y ($R_0$)")
    ax5.set_aspect("equal")

    #save fig
    if save:
        fig5.savefig(f"{save_dir}/wall64_{wall_length_float:.2f}_quiver_{iteration}.png")

if direction_histogram:
    fig6, ax6 = plt.subplots(1,1, figsize=(fig_width, fig_height),constrained_layout=True)
    histogram_data = get_histogram(wall_length,iteration)

    cax6 = ax6.imshow(histogram_data.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    if wall_length_float != 0:
        ax6 = plot_x_wall(ax6,wall_yMin=wall_yMin,wall_yMax=wall_yMax, 
                          boundary=False,
                          wall_color="r",
                          walpha=1,
                          lw=0.75,
                          linestyle = "--"
                          )
    ax6 = flow_plot(wall_length, iteration, ax = ax6, type = "stream", phase = "steady", 
                    density = 0.4,
                    color = "black",
                   )
    ax6.set_xlabel(r"x ($R_0$)")
    ax6.set_ylabel(r"y ($R_0$)")
    ax6.set_aspect("equal")
    # ax6.legend()
    cbar6 = fig6.colorbar(cax6, ax=ax6)
    cbar6.set_label("Density")

    #save fig
    if save:
        fig6.savefig(f"{save_dir}/wall64_{wall_length_float:.2f}_histogram_stream_{iteration}.png")

if dual_hitogram:
    wanted_wall_lengths = available_walls[-2:]
    # 2 plots in one showing l=1L and 1=2/3L
    fig7, ax7 = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.5), constrained_layout=True)

    for i, wall_length in enumerate(wanted_wall_lengths):
        wall_length_float = float(wall_length)
        wall_yMin = L / 2 - wall_length_float / 2
        wall_yMax = L / 2 + wall_length_float / 2

        histogram_data = get_histogram(wall_length, iteration)
        cax7 = ax7[i].imshow(histogram_data.T, extent=[0, L, 0, L], origin="lower", cmap='rainbow', aspect='auto')
        if wall_length_float != 0:
            ax7[i] = plot_x_wall(ax7[i], wall_yMin=wall_yMin, wall_yMax=wall_yMax,
                                boundary=False,
                                wall_color="white",
                                lw=0.75,
                                walpha=1)
            ax7[i] = flow_plot(wall_length, iteration, ax=ax7[i],
                            type="stream",
                            phase="steady",
                            density=0.4,
                            color="black")
        ax7[i].set_xlabel(r"x ($R_0$)")
        ax7[i].set_aspect("equal")
        ax7[i].set_xticks(np.linspace(0, L, 5))

    ax7[0].set_yticks(np.linspace(0, L, 5))
    ax7[0].set_ylabel(r"y ($R_0$)")
    ax7[1].set_ylabel(None)
    ax7[1].set_yticks([])

    # Create a colorbar for the last subplot
    cbar7 = fig7.colorbar(cax7, ax=ax7[1], location='right', fraction=0.5)
    cbar7.set_label("Density")
    # cbar7.set_ticks()
    if save:
        fig7.savefig(f"{save_dir}/wall64_{wanted_wall_lengths[0]}_{wanted_wall_lengths[1]}_histogram_stream_{iteration}.png")

plt.show()
