import numpy as np
import os 
import matplotlib.pyplot as plt
plt.style.use("default")
from fractions import Fraction

dir = str(os.getcwd())+ "\wall implemetation\wall_size_experiment"
filenames = os.listdir(dir)
print(filenames)
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

def plot_x_wall(ax, wall_x, wall_yMin, wall_yMax,wall_color = "blue", walpha = 1, *args, **kwargs):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """

    #plot the wall
    ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall", color = wall_color, alpha = walpha, *args, **kwargs)
    return ax

def stream_plot(phase,simulation, ax=None, density = 1):
    valid_phases = ["transient", "steady"]
    if phase not in valid_phases:
        raise ValueError(f"Invalid phase: '{phase}'. Valid options are {valid_phases}.")

    X,Y = simulation["stream_boundaries"]
    _Hx_stream, _Hy_stream = simulation[f"{phase}_stream"]
    if ax==None:
        fig,ax = plt.subplots()
        
    #### STREAM PLOT ##### (currently works on a for loop without the animation)
    wall_colour = "r"
    L = simulation["L"]
    wall_length = simulation["wall length"]
    x_wall = L/2
    wall_yMin = L/2 - wall_length/2
    wall_yMax = L/2 + wall_length/2
    if wall_length !=0: 
        ax = plot_x_wall(ax,x_wall,wall_yMin,wall_yMax,"red")
        ax.legend()
    
    ax.streamplot(X,Y,_Hx_stream,_Hy_stream, density=density)  
    # ax.set_title(f"{phase.title()} Phase Stream Plot\nWall Length {wall_length:.1f}")
    print(f"{phase.title()} Phase Stream Plot\nWall Length {wall_length:.1f}")
    return ax

def reading_all(filenames, simulations, dir_starter = "WALL128", iteration = -1): #, hist = True, stream = True):

    if iteration==-1:
        ending = ""
    else:
        ending = f"_{iteration}"

    for item in filenames:        
        if item.split("_")[0] == dir_starter:
            data = {"wall length": 0}
            folder = dir + "/" + item
            wall_length = item.split("_")[1]
            data["wall length"] = float(wall_length)

            
            ### Loading alignment data
            alignment_data = np.load(folder + f"/alignment_{wall_length}{ending}.npz")
            data["angle"] = alignment_data["angles"]
            data["angle_std"] = alignment_data["std"]

            for phase in ["transient", "steady"]:
                ### Loading streamplot data
                stream_data = np.load(folder + f"/{phase}_stream_plot_{wall_length}{ending}.npz")
                stream_x = stream_data["X"]
                stream_y = stream_data["Y"]
                stream_boundaries = [stream_x, stream_y]
                data[f"{phase}_stream"] = [stream_data[stream_data.files[2]],stream_data[stream_data.files[3]]]
                data["stream_boundaries"] = stream_boundaries
                
                ### Loading histogram data
                histogram_data = np.load(folder + f"/{phase}_histogram_data_{wall_length}{ending}.npz")
                data[f"{phase}_histogram"] = histogram_data["hist"]

            ### Loading summary data
            summary_file = folder + f"/simulation_parameters_{wall_length}.txt"
            summary_data = read_summary_file(summary_file)
            data["steps"] = summary_data['Total number of steps']
            data["alignment_average"] = summary_data['Alignment average steps']
            data["rho"] = summary_data["Density (rho)"]
            data["eta"] = summary_data["Noise/randomness (eta)"]
            data["L"] = summary_data["Size of box (L)"]

            # Saving final positions & orientations
            try:
                # Attempt to load the final state file
                final_state = np.load(folder + f"/finalstate_{wall_length}{ending}.npz")
                data["final_positions"] = final_state["Positions"]
                data["final_orientations"] = final_state["Orientation"]
            except FileNotFoundError as e:
                # Print the error message and continue execution
                print(f"Error: {e}. Skipping file loading and continuing...")

            simulations.append(data)
    return simulations


simulations = reading_all(filenames, [], file_starter, iteration= 5)
# print(len(simulations))
i = 2

text_width = 5.5
fig_width = text_width
fig_height = 0.75* fig_width 

width_2_subplot = fig_width/2 + 1
height_2_subplot = 0.75*width_2_subplot
height_cbar_2_subplot = 0.85*width_2_subplot

# total_height = 0.75*fig_width

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})
#### ALIGNMENT PLOTS ####
if alignment:
    
    fig, ax = plt.subplots(figsize=(text_width-1, fig_height-1)) #
    fig2,ax2 = plt.subplots(figsize=(text_width-1, fig_height-1))
    lines = []
    lines2 = []
    wall_lengths = []

    for simulation in simulations:
        angle = simulation["angle"]
        steps = simulation["steps"]
        alignment_average_frames = simulation["alignment_average"]
        std_angles = np.where(simulation["angle_std"] < 0, np.nan, simulation["angle_std"])
        

        times = np.arange(0,steps +1, alignment_average_frames)
        line = ax.plot(times, angle)[0]
        # line2 = ax2.plot(times, std_angles)
        # lines2.append(line2)
        lines.append(line)
        wall_lengths.append(simulation["wall length"])

    rho = simulations[0]["rho"]
    eta = simulations[0]["eta"]

    # Sort the lines based on wall lengths
    sorted_indices = np.argsort(wall_lengths)
    sorted_lines = [lines[i] for i in sorted_indices]
    sorted_lines2 = [lines[i] for i in sorted_indices]
    sorted_labels = [rf"$l: {Fraction(wall_lengths[i]/L).limit_denominator()}L$" for i in sorted_indices]

    ax.set_ylim(-3.22,3.22)
    ax.legend(sorted_lines, sorted_labels, frameon= False)
    ax.plot([0,times.max()],[-np.pi, -np.pi], linestyle = "-.", color = "grey", alpha = 0.4) ## Lower angle limit
    ax.plot([0,times.max()],[np.pi, np.pi], linestyle = "-.", color = "grey", alpha = 0.4) ## Upper angle limit
    ax.plot([0,times.max()],[np.pi/2, np.pi/2], linestyle = "dotted", color = "grey", alpha = 0.4) ## Upper angle limit
    ax.plot([0,times.max()],[-np.pi/2, -np.pi/2], linestyle = "dotted", color = "grey", alpha = 0.4) ## Upper angle limit
    ax.set_xlabel("time")
    ax.set_ylabel(r'$\langle \theta \rangle$')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

### AVERAGING STD ALIGNMENTS OVER 6 SIMULATIONS
simulations = []
for J in range(6):
    # print(i)
    simulations = reading_all(filenames, simulations,file_starter , iteration=J)

# Dictionary to store std angles for each wall length
wall_std_dict = {}
wall_lengths = []
wall_hist_dict = {}
# Go through all three iterations
for iteration in [0,1,2,3,4,5]:
    for simulation in simulations[6*iteration:6*(1+iteration)]:
        wall_length = simulation["wall length"]
        steps = simulation["steps"]
        alignment_average_frames = simulation["alignment_average"]
        std_angles = np.where(simulation["angle_std"] < 0, np.nan, simulation["angle_std"])
        times = np.arange(0, steps + 1, alignment_average_frames)
        
        # Initialize list for this wall length if not exists
        if wall_length not in wall_std_dict:
            wall_std_dict[wall_length] = []
            wall_hist_dict[wall_length] = simulation["steady_histogram"]
        
        # Append the std_angles for this iteration
        wall_std_dict[wall_length].append(std_angles)
        wall_hist_dict[wall_length] += simulation["steady_histogram"]

for wall_length, std_list in wall_std_dict.items():
    
    # Convert to numpy array and calculate mean across iterations
    avg_std = np.mean(std_list, axis=0)
    
    # Convert wall_length to a fraction
    wall_label_fraction = Fraction(wall_length / 128.0).limit_denominator()
    wall_label = f"wall length: {wall_label_fraction}"
    
    if alignment:
        line2 = ax2.plot(times, avg_std, label=wall_label)[0]
        lines2.append(line2)
    
    wall_lengths.append(wall_length / 128.0)

if alignment:
    ax2.legend(sorted_lines2, sorted_labels, frameon=False)
    ax2.set_xlabel("time")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel(r"$\sigma$ ($\theta$)")
    ax2.set_ylim(0.4,3.2)
# ax2.set_title("Variation of alignment Viscek Particles with a Wall")

#### STREAM PLOTS #####
## Take a while to load
if stream_plots:
    # First figure - Transient state
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_width))
    ax3 = stream_plot("transient", simulations[i], ax3, 1)
    ax3.set_xlim(0, L)
    ax3.set_ylim(0, L)

    # Second figure - Steady state
    fig4, ax4 = plt.subplots(figsize=(fig_width, fig_width))
    ax4 = stream_plot("steady", simulations[i], ax4, 0.5)
    ax4.set_xlim(0, L)
    ax4.set_ylim(0, L)

    wall_length = simulations[i]["wall length"]
    zoom_distance = wall_length/2 +2
    cmap = "rainbow"


#### DENSITY HISTOGRAMS ####
i =2
if histograms:
    # plt.rcParams.update({'font.size': 16}) 
    wall_length = wall_lengths[i]*128
    print(Fraction(wall_lengths[i]).limit_denominator())
    hist = wall_hist_dict[wall_length]
    fig6, ax6 = plt.subplots(figsize=(fig_width, fig_width))
    #### _, ax6 = hist_2D_plot("steady", simulations[i], fig6, ax6, w_alpha=0, cmap=cmap, wall_color="pink", linestyle = "dashdot") ### This does it for 1 simulaiton only
    cax = ax6.imshow(hist.T, extent=[0,L,0,L], origin="lower", cmap='rainbow', aspect='auto') # This summs all simulaitons to find an average
    cbar = fig6.colorbar(cax, ax=ax6, ticks=[])  # Removes ticks and labels in one line
    cbar.set_label('Density')
    cbar.ax.tick_params(labelsize=0)   # Remove colorbar tick labels
    lower =  36
    upper = 88
    ax6.set_xlim(lower, upper)
    ax6.set_ylim(lower, upper)
    ax6.set_aspect('equal')  # Ensure square aspect ratio
    ax6.set_xticks(np.linspace(lower,upper,5))
    ax6.set_yticks(np.linspace(lower,upper,5))
    plt.tight_layout()

#### PLOTTING FINAL POSITIONS ####
if final_positions:
    fig7, ax7 = plt.subplots(figsize = (fig_width, fig_width))
    i = 16
    positions = simulations[i]["final_positions"]
    orientation = simulations[i]["final_orientations"]
    wall_length = simulations[i]["wall length"]
    x_wall = simulations[i]["L"]/2
    wall_yMin = simulations[i]["L"]/2 - wall_length/2
    wall_yMax = simulations[i]["L"]/2 + wall_length/2
    qv = ax7.quiver(positions[:,0], positions[:,1], np.cos(orientation), np.sin(orientation), orientation, clim = [-np.pi, np.pi], cmap = "hsv")
    if wall_length != 0:
        ax7 = plot_x_wall(ax7, x_wall, wall_yMin, wall_yMax, "red",walpha= 1)
        ax7.legend(loc = "upper right")
    ax7.set_xticks([0,simulations[i]["L"]/4, simulations[i]["L"]/2, 3*simulations[i]["L"]/4, simulations[i]["L"]])
    ax7.set_yticks([0,simulations[i]["L"]/4, simulations[i]["L"]/2, 3*simulations[i]["L"]/4, simulations[i]["L"]])
    ax7.set_xlim(0, simulations[i]["L"])
    ax7.set_ylim(0, simulations[i]["L"])
    plt.margins(0)  # Remove padding
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)


plt.show()

