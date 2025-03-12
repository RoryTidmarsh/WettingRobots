import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "/wall implemetation/arc_analysis"
filenames = os.listdir(dir)
dir_starter = "arc64"
print(filenames)
cmap = "hsv"
L = 50
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

R = summary_data["Radius"]
L = summary_data['Size of box (L)']
eta = summary_data['Noise/randomness (eta)']
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
def plot_arc(ax, wall_color = "r", boundary = True, walpha = 1):
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
    ax.plot(x,y, color = wall_color, alpha= walpha, lw=0.5)

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
        cax = ax[i].imshow(histogram_data.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
        
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
    # cax = ax.imshow(histogram_data, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    # cbar= fig.colorbar(cax, ax=ax)
    # cbar.set_label("Density")

def stream_plot(angle, iteration, phase = "steady", ax=None, density = 1, *args, **kwargs):
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
    
    #Only plot inside the box, (not into and out of the walls)
    avg_vx_inner = avg_vx[1:-1,1:-1]
    avg_vy_inner = avg_vy[1:-1,1:-1]
    X_inner = X[1:-1,1:-1]*L
    Y_inner = Y[1:-1,1:-1]*L

    if ax==None:
        fig,ax = plt.subplots()
    ax.streamplot(X_inner,Y_inner,avg_vx_inner.T,avg_vy_inner.T, density=density, *args, **kwargs)

    return ax


def flow_plot(angle, iteration, phase = "steady", ax=None, density = 1, *args, **kwargs):
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
    
    return ax

iteration = 0
if quiver_plot:
    ncols  = 2
    angle = available_angles[1]
    # fig2, ax2 = plt.subplots(1,ncols, figsize=(fig_width,fig_width), constrained_layout=True)
    # for i in range(ncols):
    #     ax2[i] = flow_plot(angle, 1,ax=ax2[i])
    #     ax2[i] = plot_arc(ax2[i], boundary=False)
    #     ax2[i].set_title(f"Iteration {i}")
    #     ax2[i].set_aspect('equal')

    fig2,ax2 = plt.subplots(1,1, figsize=(fig_width,fig_width), constrained_layout=True)
    ax2 = flow_plot(angle, iteration=iteration,ax=ax2)
    ax2 = plot_arc(ax2, boundary=False)
    hist = get_histogram(angle, iteration=iteration)
    ax2.imshow(hist.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    ax2.set_aspect('equal')


        
if stream:
    ncols = 2
    angle = available_angles[5]
    # fig3, ax3 = plt.subplots(1,ncols, figsize=(fig_width,fig_width), constrained_layout=True)
    # for i in range(ncols):
    #     ax3[i] = stream_plot(angle, 1,ax=ax3[i], density= 0.5)
    #     ax3[i] = plot_arc(ax3[i], boundary=False)
    #     ax3[i].set_title(f"Iteration {i}")
    #     ax3[i].set_aspect('equal')
    fig3, ax3 = plt.subplots(1,1, figsize=(fig_width,fig_width), constrained_layout=True)
    ax3 = stream_plot(angle, iteration=iteration,ax=ax3,
                      density=0.5,
                      color = "black",
                      integration_direction='both',
                      broken_streamlines=True,
                      )
    ax3 = plot_arc(ax3, boundary=False)
    hist = get_histogram(angle, iteration=iteration)
    ax3.imshow(hist.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto')
    ax3.set_aspect('equal')
    

plt.show()