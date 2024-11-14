import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numba
# parameters
L = 10.0 # size of box
rho = 3 # density
N = int(rho * L**2) # number of particles
r0 = 0.65 # interaction radius
deltat = 1.0 # time step
velocity_factor = 0.2
v0 = r0 / deltat * velocity_factor # velocity
iterations = 400 # animation frames
eta = 0#0.3 # noise/randomness
max_num_neighbours= N


#Defining parameters for a wall only in the x direction.
wall_x = 5.
wall_yMin = 1.
wall_yMax = 9.
wall_distance = r0/3
wall_turn = np.deg2rad(110)
turn_factor = 1.

#Defining parameters for a rectangle
x_min, x_max, y_min, y_max = 4.,6.,4.,6.

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) 

av_frames_angles = 10
num_frames_av_angles = np.empty(av_frames_angles)
t = 0
@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles*1.0j)))
average_angles = [average_angle(positions)]
# average_angles2 = []

###Average displacement in a 2D histogram over time
bins = int(L*5/r0)
hist, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= bins, density = False)

@numba.njit
def x_wall_filter(x_pos,y_pos):
    """Finds the distance of the arrow to the wall.

    Args:
        x_pos (float): x_position of 1 particle
        y_pos (float): y_position of 1 particle
    Returns:
        distance_to_wall: distance of the particle to the wall
    """
    if y_pos > wall_yMax:
        #particle above the wall
        distance_to_wall = np.sqrt((x_pos-wall_x)**2 + (y_pos-wall_yMax)**2)
    elif y_pos < wall_yMin:
        #particle below the wall
        distance_to_wall = np.sqrt((x_pos-wall_x)**2 + (y_pos-wall_yMin)**2)
    else:
        #particle level with the wall
        distance_to_wall = np.abs(x_pos-wall_x)
    return distance_to_wall

@numba.njit
def varying_angle_turn(x_pos, y_pos, turn_factor):
    """Tells the particle how much to turn when it is within the boundary as a function of its distance to the boundary

    Args:
        dist (float): distance the particle is to the wall
        turn_factor (float): how quickly the particle should turn when approaching the wall

    Returns:
        turn_angle: angle for the particle to turn
    """
    global wall_distance, wall_turn

    dist = x_wall_filter(x_pos, y_pos)
    
    if (dist<wall_distance)&(y_pos>y_min)&(y_pos<y_max): #close to wall verically 
        turn_angle =  1/(turn_factor*(x_pos-wall_x))+0j

    elif (dist<wall_distance)&(y_pos>y_max): #close to wall top semi circle
        turn_angle = 1/(turn_factor*(x_pos - wall_x)) + 1j/(turn_factor*(y_pos - y_max))

    elif (dist<wall_distance)&(y_pos<y_min): #close to wall bottom semi circle
        turn_angle = 1/(turn_factor*(x_pos - wall_x)) + 1j/(turn_factor*(y_pos - y_min))
    else:
        turn_angle = 0
    # turn_angle = np.where(dist < wall_distance, wall_turn * np.exp(-turn_factor * dist), 0)
    return turn_angle

def plot_x_wall_boundary(ax, wall_color = "blue"):
    """plots the boundary based on the initial dimensions of the wall set.

    Args:
        ax (matplotlib axis): input axis for the wall to be plotted onto

    Returns:
        ax: plot including the wall.
    """
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
    ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall", color = wall_color)
    return ax

def position_filter(positions, filter_func):
    """Adjusts the initial positions so they are no longer within the boundary.

    Args:
        positions (numpy array): array of positions
        filter_func (function): function that calculates the distance to the wall

    Returns:
        positions
    """
    for i in range(len(positions)):
        # Check if the particle is too close to the wall
        while filter_func(positions[i][0], positions[i][1]) <= wall_distance:
            # Regenerate position until it is far enough from the wall
            positions[i] = np.random.uniform(0, L, size=(1, 2))
    return positions

positions = position_filter(positions, x_wall_filter)

@numba.njit
def update(positions, angles, func):
    # empty arrays to hold updated positions and angles
    new_positions = np.empty_like(positions)
    new_angles = np.empty_like(angles)
    neigh_angles = np.empty(max_num_neighbours)
    # loop over all particles
    for i in range(N):
        # list of angles of neighbouring particles
        # neighbour_angles = []
        count_neigh = 0

        # distance to other particles
        for j in range(N):
            distance = np.linalg.norm(positions[i] - positions[j])
            # if within interaction radius add angle to list
            if (distance < r0) & (distance != 0):
                # neighbour_angles[].append(angles[j])
                neigh_angles[count_neigh] = angles[j]
                count_neigh += 1
        x_pos = positions[i,0]
        y_pos = positions[i,1]
         
        distance_to_wall = func(x_pos, y_pos) 
        # if there are neighbours, calculate average angle and noise/randomness       
        # if neighbour_angles:
        wall_turn = varying_angle_turn(x_pos, y_pos,turn_factor=turn_factor)
        noise = eta * np.random.uniform(-np.pi, np.pi)
        if count_neigh > 0:
            average_angle = np.sum(np.exp(neigh_angles[:count_neigh])*1.0j)
            new_complex = average_angle + wall_turn
            # average_angle = np.angle(np.sum(np.exp((neigh_angles[:count_neigh])*1.0j)))
            
            # if angles[i] <= 0:
            #     new_angles[i] = average_angle + wall_turn + noise
            # else:
            #     new_angles[i] = average_angle + wall_turn + noise
            
        else:
            new_complex = np.exp(angles[i]*1j) + wall_turn
            # # if no neighbours, keep current angle unless close to wall
            # if angles[i] <= 0:
            #     new_angles[i] = angles[i] - wall_turn + noise
            # else:
            #     new_angles[i] = angles[i] + wall_turn + noise
            # # Make the particle turn around based on the wall_turn parameter
        new_angles[i] = noise + np.angle(new_complex)
        
        # update position based on new angle
        # new position from speed and direction   
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        # boundary conditions of bo
        new_positions[i] %= L

    return new_positions, new_angles

def animate(frames):
    print(frames)
    global positions, angles, t, num_frames_av_angles, hist
    
    new_positions, new_angles = update(positions, angles,x_wall_filter)
    
    # Store the new angles in the num_frames_av_angles array
    # average_angles2.append(average_angle(new_angles))
    num_frames_av_angles[t] = average_angle(new_angles)
    if t == av_frames_angles - 1:  # Check if we've filled the array
        average_angles.append(average_angle(num_frames_av_angles))
        t = 0  # Reset t
        num_frames_av_angles = np.empty(av_frames_angles)  # Reinitialize the array
    else:
        t += 1  # Increment t
        
    #Add positions to the 2D histogram
    hist += np.histogram2d(new_positions[:, 0], new_positions[:,1], bins= [xedges,yedges], density = False)[0]
    # Update global variables
    positions = new_positions
    angles = new_angles

    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    return qv,
 
fig, ax = plt.subplots(figsize = (6, 6))   

ax = plot_x_wall_boundary(ax)
ax.set_title(f"{N} particles, turning near a wall. Varying angle with wall distance.")

qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")

ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
ax.legend(loc = "upper right")
# ani.save(f'code/wall implemetation/figures/Vicek_varying_wall_turn_p={rho:.2f}.gif', writer='imagemagick', fps=30)
plt.show()

fig, ax2 = plt.subplots()
times = np.arange(0,len(average_angles))*av_frames_angles
ax2.plot(times, average_angles, label = "10 frame average")
# ax2.plot(np.arange(len(average_angles2)), average_angles2, label = "1 frame")
ax2.set_xlabel("Time")
ax2.set_ylabel("Angle (radians)")
ax2.set_title("Alignment, averaging from different number of frames.")
ax2.legend(loc = "upper left")

ax2.plot([0,times.max()],[-np.pi, -np.pi], linestyle = "--", color = "grey", alpha = 0.4)
ax2.plot([0,times.max()],[np.pi, np.pi], linestyle = "--", color = "grey", alpha = 0.4)
ax2.grid()
# plt.show()

hist_normalised = hist.T/sum(hist)
# After the animation and histogram calculations
fig, ax3 = plt.subplots(figsize=(6, 6))
# Use imshow to display the normalized histogram
cax = ax3.imshow(hist_normalised, extent=[0, L, 0, L], origin='lower', cmap='rainbow', aspect='auto')
ax3 = plot_x_wall_boundary(ax3, "red")
ax3.set_xlabel("X Position")
ax3.set_ylabel("Y Position")
ax3.set_title(f"2D Histogram of Particle Positions over {len(times)*10} timesteps.")
ax3.legend()
# Add a colorbar for reference
fig.colorbar(cax, ax=ax3, label='Density')
plt.show()