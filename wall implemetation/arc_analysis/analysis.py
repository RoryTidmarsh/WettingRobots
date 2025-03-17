import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "/wall implemetation/arc_analysis"
filenames = os.listdir(dir)
dir_starter = "arc64"
print(filenames)
cmap = "viridis"
L = 64
r0 = 1

## Figure information
text_width = 3.25  # inches (single column width)
fig_width = text_width
fig_height = 0.8 * fig_width  # for standard plots
width_2_subplot = fig_width/2 + 0.25  # for side-by-side subplots
height_2_subplot = 0.75 * width_2_subplot
height_cbar_2_subplot = 0.75 * width_2_subplot
plt.rcParams.update({
    'font.size': 8,
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
})

histograms = False
stream = True
quiver_plot = False

save = True

def read_summary_file(filepath):
    summary_data = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Split the line into key and value
            key, value = line.strip().split(': ')
            # Store the key-value pair in the dictionary
            summary_data[key] = float(value) if '.' in value or value.isdigit() else value
    return summary_data

filepath = dir + "\\arc64_0.00_10000\simulation_parameters_0.0.txt"
summary_data = read_summary_file(filepath)

R = float(summary_data["Radius"])
L = float(summary_data['Size of box (L)'])
eta = float(summary_data['Noise/randomness (eta)'])
r0 = float(summary_data['Interaction radius (r0)'])
rho = float(summary_data['Density (rho)'])
center_x = float(summary_data["center_x"])
center_y = float(summary_data["center_y"])

def get_params():
    available_angles = set() # multiples of pi
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter:
            angle = exp_item.split("_")[1]
            available_angles.add(angle)
        
    return sorted(list(available_angles))

available_angles = get_params()
print(available_angles)

def get_histogram(angle, iteration):
    for exp_item in os.listdir(dir):
        if exp_item.split("_")[0] == dir_starter and exp_item.split("_")[1] == angle:
            folder_path = os.path.join(dir, exp_item)
            for item in os.listdir(folder_path):
                if item.endswith(f"{iteration}.npz") and "steady_histogram" in item:
                    histogram_path = os.path.join(folder_path, item)
                    histogram_data = np.load(histogram_path)["hist"]
                    return histogram_data
def plot_arc(ax, wall_color = "r", boundary = True, walpha = 1, *args, **kwargs):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
    global center_x,center_y, start_angle, radius, angle, r0
    radius = R
    wall_distance = r0
    arc_angle = float(angle)*np.pi
    start_angle = -arc_angle/2
    thetas = np.linspace(start_angle, start_angle+arc_angle,200)
    x = center_x + radius*np.cos(thetas)
    y = center_y + radius*np.sin(thetas)
    ax.plot(x,y, color = wall_color, alpha= walpha, lw=0.5, *args, **kwargs)

    if boundary==True:
        #Plotting the far and near arcs
        x_far = center_x + (radius+wall_distance)*np.cos(thetas)
        y_far = center_y + (radius+wall_distance)*np.sin(thetas)
        ax.plot(x_far,y_far, 'b--', lw=2)

        x_close = center_x + (radius-wall_distance)*np.cos(thetas)
        y_close = center_y + (radius-wall_distance)*np.sin(thetas)
        ax.plot(x_close,y_close, 'b--', lw=2)

        # Plot full circles at each endpoint
        end1_x = center_x + radius * np.cos(start_angle)
        end1_y = center_y + radius * np.sin(start_angle)
        end2_x = center_x + radius * np.cos(start_angle + arc_angle)
        end2_y = center_y + radius * np.sin(start_angle + arc_angle)
        circle_thetas = np.linspace(0, 2*np.pi, 100)
        # First endpoint circle
        ax.plot(end1_x + wall_distance * np.cos(circle_thetas),
                end1_y + wall_distance * np.sin(circle_thetas),
                'b--', lw=2)
        # Second endpoint circle
        ax.plot(end2_x + wall_distance * np.cos(circle_thetas),
                end2_y + wall_distance * np.sin(circle_thetas),
                'b--', lw=2)
    return ax
                
if histograms:

    fig, ax = plt.subplots(1,3, figsize=(7, 1.5), constrained_layout=True)
    angle = available_angles[1]
    
    start_angle = - float(angle)/2
    for i in range(3):
        histogram_data = get_histogram(angle, i)
        cax = ax[i].imshow(histogram_data.T, extent=[0,L,0,L], origin="lower", cmap=cmap, aspect='auto')
        
        ax[i] = plot_arc(ax[i], boundary=False)
        ax[i].set_title(f"Iteration {i}")
        ax[i].set_aspect('equal')
    ax[1].spines['left'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    # histogram_data = get_histogram(angle, 1)
    cbar= fig.colorbar(cax, ax=ax[-1])
    cbar.set_label("Density")
    # fig.tight_layout()
    # cax = ax.imshow(histogram_data, extent=[0,L,0,L], origin="lower", cmap=cmap, aspect='auto')
    # cbar= fig.colorbar(cax, ax=ax)
    # cbar.set_label("Density")

def get_simulation_region(angle):
    """Get the relevant region of the simulation for plotting based on arc angle"""
    # For full circle (2*pi), we need to focus on just the circle region
    if angle == "2.00":  # This assumes 2.00 represents 2.00*pi
        return {
            'x_min': center_x - R - 2,
            'x_max': center_x + R + 2,
            'y_min': center_y - R - 2,
            'y_max': center_y + R + 2
        }
    else:
        # For partial arcs, use the full box
        return {
            'x_min': 0,
            'x_max': L,
            'y_min': 0,
            'y_max': L
        }

def mask_data(X, Y, X_hist, Y_hist, center_x, center_y, radius):
    """Apply circular mask to velocity data, with 0s outside the radius"""
    # Calculate distances from center
    dx = X*L - center_x     # X and Y are normalised, so multiply by L to get real distances
    dy = Y*L - center_y
    distances_squared = dx**2 + dy**2
    
    # Create mask and apply it
    mask = distances_squared >= radius**2    
    masked_X_hist = np.where(mask, 0, X_hist)
    masked_Y_hist = np.where(mask, 0, Y_hist)
    
    return masked_X_hist, masked_Y_hist

def stream_plot(angle, iteration, phase="steady", ax=None, density=1, radius = R, *args, **kwargs):
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
    
    X, Y = simulation["X"], simulation["Y"]
    _Hx_stream, _Hy_stream = simulation["X_hist"], simulation["Y_hist"]
    try:
        counts = simulation["counts"]
    except:
        counts = np.ones_like(_Hx_stream)

    avg_vx = np.zeros_like(X)
    avg_vy = np.zeros_like(Y)
    avg_vx[counts > 0] = _Hx_stream[counts > 0] / counts[counts > 0]
    avg_vy[counts > 0] = _Hy_stream[counts > 0] / counts[counts > 0]

    # Mask the data to ensure the streamlines are only inside the circle
    avg_vx, avg_vy = mask_data(X, Y, avg_vx, avg_vy, center_x, center_y, radius = radius)
    
    # Only plot inside the box, (not into and out of the walls)
    avg_vx_inner = avg_vx[1:-1, 1:-1]
    avg_vy_inner = avg_vy[1:-1, 1:-1]
    X_inner = X[1:-1, 1:-1] * L
    Y_inner = Y[1:-1, 1:-1] * L

    if ax is None:
        fig, ax = plt.subplots()
    
    # Get the appropriate region for this arc angle
    region = get_simulation_region(angle)
    
    # Set the axis limits to match the simulation region
    ax.set_xlim(region['x_min'], region['x_max'])
    ax.set_ylim(region['y_min'], region['y_max'])
    
    # Create the stream plot
    ax.streamplot(X_inner, Y_inner, avg_vx_inner.T, avg_vy_inner.T, density=density, *args, **kwargs)

    return ax


def flow_plot(angle, iteration, phase = "steady", ax=None, density = 1, radius = R, *args, **kwargs):
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
    # counts = np.ones_like(_Hx_stream)

    avg_vx = np.zeros_like(X)
    avg_vy = np.zeros_like(Y)
    avg_vx[counts > 0] = _Hx_stream[counts > 0] / counts[counts > 0]
    avg_vy[counts > 0] = _Hy_stream[counts > 0] / counts[counts > 0]

    # Mask the data to ensure the streamlines are only inside the circle
    avg_vx, avg_vy = mask_data(X, Y, avg_vx, avg_vy, center_x, center_y, radius = radius)
    
    #Only plot inside the box, (not into and out of the walls)
    avg_vx_inner = avg_vx[1:-1,1:-1]
    avg_vy_inner = avg_vy[1:-1,1:-1]
    X_inner = X[1:-1,1:-1]*L
    Y_inner = Y[1:-1,1:-1]*L

        # Downsample the data based on the density parameter
    if density < 1:
        step = int(1 / density)
        X_inner = X_inner[::step, ::step]
        Y_inner = Y_inner[::step, ::step]
        avg_vx_inner = avg_vx_inner[::step, ::step]
        avg_vy_inner = avg_vy_inner[::step, ::step]

    if ax is None:
        fig, ax = plt.subplots()
    ax.quiver(X_inner, Y_inner, avg_vx_inner.T, avg_vy_inner.T, scale=0.2, scale_units='xy', *args, **kwargs)
    
    region = get_simulation_region(angle)
    
    # Set the axis limits to match the simulation region
    ax.set_xlim(region['x_min'], region['x_max'])
    ax.set_ylim(region['y_min'], region['y_max'])
    
    ax.quiver(X_inner, Y_inner, avg_vx_inner.T, avg_vy_inner.T, scale=0.2, scale_units='xy', *args, **kwargs)
    
    return ax

iteration = 0
if quiver_plot:
    ncols  = 2
    angle = available_angles[-1]
    # fig2, ax2 = plt.subplots(1,ncols, figsize=(fig_width,fig_width), constrained_layout=True)
    # for i in range(ncols):
    #     ax2[i] = flow_plot(angle, 1,ax=ax2[i])
    #     ax2[i] = plot_arc(ax2[i], boundary=False)
    #     ax2[i].set_title(f"Iteration {i}")
    #     ax2[i].set_aspect('equal')

    fig2,ax2 = plt.subplots(1,1, figsize=(fig_width,fig_width*0.7), constrained_layout=True)
    ax2 = flow_plot(angle, iteration=iteration,ax=ax2, radius = R*0.95)
    ax2 = plot_arc(ax2, boundary=False)
    hist = get_histogram(angle, iteration=iteration)
    cax2 = ax2.imshow(hist.T, extent=[0,L,0,L], origin="lower", cmap=cmap, aspect='auto')
    cbar2 = fig2.colorbar(cax2, ax=ax2)
    cbar2.set_label("Density")
    ax2.set_aspect('equal')

    if save:
        fig2.savefig(f"figures/arc_flow_{float(angle):.2f}.png", dpi=300)

iteration = 3
if stream:
    angle = available_angles[1]  # This is the full circle case
    fig3, ax3 = plt.subplots(1, 1, figsize=(fig_width, fig_width*0.7), constrained_layout=True)
    # Get the histogram first
    hist = get_histogram(angle, iteration=iteration)
    hist /= hist.sum()
    # Get the region info
    region = get_simulation_region(angle)
    # Plot the histogram with the correct extent
    im = ax3.imshow(
        hist.T, 
        extent=[0, L, 0, L], 
        origin="lower", 
        cmap=cmap, 
        aspect='auto'
    )
    # Set the axis limits based on region
    ax3.set_xlim(region['x_min'], region['x_max'])
    ax3.set_ylim(region['y_min'], region['y_max'])
    # Add the stream plot
    if angle == "2.00":
        radius = 0.91*R
    else:
        radius = 2*L
    ax3 = stream_plot(
        angle, 
        iteration=iteration,
        ax=ax3,
        density=0.45,
        color="white",
        radius=radius,
        integration_direction='both',
        broken_streamlines=True,
    )   
    # Add the arc
    ax3 = plot_arc(ax3, boundary=False, linestyle = "--", label = "Barrier")
    # ax3.set_xlim(20,64)
    # ax3.set_ylim(10,54)
    ax3.set_aspect('equal')
    ax3.legend(loc = "upper left")
    # Add the colorbar
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label("Density")

    

    if save:
        fig3.savefig(f"figures/arc_hist_stream_{float(angle):.2f}.png", dpi=300)
    

plt.show()