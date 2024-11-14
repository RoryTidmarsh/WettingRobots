import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
import os
# parameters
L = 10.0 # size of box
rho = 0.7 # density
N = int(rho * L**2) # number of particles
r0 = 1.0 # interaction radius
deltat = 1.0 # time step
factor = 0.2
v0 = r0 / deltat * factor # velocity
iterations = 100 # animation frames
eta = 0.15 # noise/randomness
max_num_neighbours = N # it could be less...

# initialise positions and angles
positions = np.random.uniform(0, L, size = (N, 2))
angles = np.random.uniform(-np.pi, np.pi, size = N) # from 0 to 2pi rad

savedir = "npzfiles/"

### Alignment over time
av_frames_angles = 10
num_frames_av_angles = np.empty(av_frames_angles)
num_frames_av_angles2 = np.empty(av_frames_angles)
t = 0
@numba.njit
def average_angle(new_angles):
    return np.angle(np.sum(np.exp(new_angles * 1.0j)))

average_angles = [average_angle(positions)]

###Average displacement in a 2D histogram over time
bins = int(L/(r0/2))
hist, xedges, yedges = np.histogram2d(positions[:, 0], positions[:,1], bins= bins, density = False)

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")
delete_files_in_directory(savedir)
# Function to output parameters to a file
def output_parameters(filename=f'{savedir}simulation_parameters.txt'):
    with open(filename, 'w') as f:
        f.write(f"Size of box (L): {L}\n")
        f.write(f"Density (rho): {rho}\n")
        f.write(f"Number of particles (N): {N}\n")
        f.write(f"Interaction radius (r0): {r0}\n")
        # f.write(f"Time step (deltat): {deltat}\n")
        # f.write(f"Velocity (v0): {v0}\n")
        # f.write(f"Number of iterations: {iterations}\n")
        f.write(f"Noise/randomness (eta): {eta}\n")
        f.write(f"Max number of neighbours: {max_num_neighbours}\n")
        f.write(f"Total number of steps: {step} \n")
@numba.njit
def update(positions, angles):
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
         
        # if there are neighbours, calculate average angle and noise/randomness       
        # if neighbour_angles:
        noise = eta * np.random.uniform(-np.pi, np.pi)
        if count_neigh > 0:
            average_angle = np.angle(np.sum(np.exp((neigh_angles[:count_neigh])*1.0j)))
            
            new_angles[i] = average_angle + noise # updated angle with noise
        else:
            # if no neighbours, keep current angle
            new_angles[i] = angles[i] + noise
        
        # update position based on new angle
        # new position from speed and direction   
        new_positions[i] = positions[i] + v0 * np.array([np.cos(new_angles[i]), np.sin(new_angles[i])]) * deltat
        # boundary conditions of bo
        new_positions[i] %= L
    # t+=1
    # average_angles[t+1] = np.sum(new_angles)
    return new_positions, new_angles
step = 0
def animate(frames):
    print(frames)
    global positions, angles, t, num_frames_av_angles, hist, step
    
    new_positions, new_angles = update(positions, angles)
    
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
    hist += np.histogram2d(positions[:, 0], positions[:,1], bins= [xedges,yedges], density = False)[0]
        
    # Update global variables
    positions = new_positions
    angles = new_angles

    step +=1
    np.savez_compressed(f'{savedir}Viscek_Simulation_{step}.npz', positions=np.array(positions, dtype = np.float16), angles=np.array(angles, dtype = np.float16))

    # Update the quiver plot
    qv.set_offsets(positions)
    qv.set_UVC(np.cos(new_angles), np.sin(new_angles), new_angles)
    return qv,

fig, ax = plt.subplots(figsize = (6, 6))   
qv = ax.quiver(positions[:,0], positions[:,1], np.cos(angles), np.sin(angles), angles, clim = [-np.pi, np.pi], cmap = "hsv")
ax.grid()
ani = FuncAnimation(fig, animate, frames = range(1, iterations), interval = 5, blit = True)
plt.show()

output_parameters()

# fig, ax2 = plt.subplots()
# times = np.arange(0,len(average_angles))*av_frames_angles
# ax2.plot(times, average_angles, label = "10 frames.")
# ax2.plot(np.arange(len(average_angles2)), average_angles2, label = "1 frame")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Angle (radians)")
# ax2.set_title("Alignment, averaging from different number of frames.")
# ax2.legend()
# plt.show()

# hist_normalised = hist.T/sum(hist)
# # After the animation and histogram calculations
# fig, ax3 = plt.subplots(figsize=(6, 6))
# # Use imshow to display the normalized histogram
# cax = ax3.imshow(hist_normalised, extent=[0, L, 0, L], origin='lower', cmap='hot', aspect='auto')
# ax3.set_xlabel("X Position")
# ax3.set_ylabel("Y Position")
# ax3.set_title("Normalised 2D Histogram of Particle Positions")
# # Add a colorbar for reference
# fig.colorbar(cax, ax=ax3, label='Density')
# plt.show()