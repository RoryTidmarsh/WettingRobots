import matplotlib as plt
import numpy as np
import numba

# @numba.experimental.jitclass()
class x_wall():
    """class producing the wall filters and plots.
    """
    def __init__(self, x, y_max, y_min, wall_distance):
        self.x = x
        self.y_max = y_max
        self.y_min = y_min
        self.wall_distance = wall_distance

    def plot_x_wall_boundary(self, ax):
        """plots the boundary based on the initial dimensions of the wall set.

        Args:
            ax (matplotlib axis): input axis for the wall to be plotted onto

        Returns:
            ax: plot including the wall.
        """
        ### Initialising the variables for the wall
        wall_x = self.x
        wall_distance = self.wall_distance
        wall_yMin = self.y_min
        wall_yMax = self.y_max

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
        ax.plot([wall_x,wall_x],[wall_yMin,wall_yMax], label = "wall")
        return ax
    
    # @numba.njit
    def x_wall_filter(self, x_pos,y_pos):
        """Finds the distance of the arrow to the wall.

        Args:
            x_pos (float): x_position of 1 particle
            y_pos (flaot): y_position of 1 particle
        Returns:
            distance_to_wall: distance of the particle to the wall
        """
        wall_x = self.x
        wall_yMin = self.y_min
        wall_yMax = self.y_max
        
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
    
class rectangle():
    def __init__(self, x_min, x_max, y_min, y_max, wall_distance):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = np.abs(x_max- x_min)
        self.height = np.abs(y_max- y_min)
        self.wall_distance = wall_distance
        # self.rect_colour = rect_colour
        # self.border_color = border_color
    
    def plot_rectangle_boundary(self, ax, linetype = "b--", facecolor = "blue", alpha = 0.7):
        
        width = self.width
        height = self.height
        y_min = self.y_min
        y_max = self.y_max
        x_min = self.x_min
        x_max = self.x_max
        wall_distance = self.wall_distance

        ax.plot([x_max, x_min], np.array([y_min, y_min]) - wall_distance, linetype)
        ax.plot([x_max, x_min], np.array([y_max, y_max]) + wall_distance, linetype)
        ax.plot(np.array([x_min, x_min]) - wall_distance, [y_min, y_max], linetype)
        ax.plot(np.array([x_max, x_max]) + wall_distance, [y_min, y_max], linetype)

        # Define the corner angles
        theta = np.linspace(0, np.pi/2, 100)  # For a quarter circle

        # Top-left corner
        top_left_x = x_min - wall_distance * np.cos(theta)
        top_left_y = y_max + wall_distance * np.sin(theta)
        ax.plot(top_left_x, top_left_y, linetype)

        # Top-right corner
        top_right_x = x_max + wall_distance * np.cos(theta)
        top_right_y = y_max + wall_distance * np.sin(theta)
        ax.plot(top_right_x, top_right_y, linetype)

        # Bottom-right corner
        bottom_right_x = x_max + wall_distance * np.cos(theta)
        bottom_right_y = y_min - wall_distance * np.sin(theta)
        ax.plot(bottom_right_x, bottom_right_y, linetype)

        # Bottom-left corner
        bottom_left_x = x_min - wall_distance * np.cos(theta)
        bottom_left_y = y_min - wall_distance * np.sin(theta)
        ax.plot(bottom_left_x, bottom_left_y, linetype)

        rect = plt.Rectangle((x_min, y_min), height, width, facecolor = facecolor, alpha = alpha)
        ax.add_artist(rect)
        return ax
    
    # def plot_rectangle(self, ax, color = 'grey', alpha =0.7) :   
    #     # Calculate the width and height of the rectangle
    #     width = self.width
    #     height = self.height
    #     y_min = self.y_min
    #     # y_max = self.y_max
    #     x_min = self.x_min
    #     # x_max = self.x_max
    #     rect = plt.Rectangle((x_min, y_min), height, width, facecolor = color, alpha = alpha)
    #     ax.add_artist(rect)
    #     return ax
    
    # @numba.njit
    def rectangle_wall_filter(self, x_pos, y_pos):
        """
        Finds the distance of the particle to the nearest wall of a rectangle.

        Args:
            x_pos (float): x position of the particle
            y_pos (float): y position of the particle
        x_min, x_max, y_min, y_max
        Returns:
            float: distance of the particle to the nearest wall
        """
        # width = self.width
        # height = self.height
        y_min = self.y_min
        y_max = self.y_max
        x_min = self.x_min
        x_max = self.x_max
        # If the particle is outside the rectangle, calculate the Euclidean distance to the nearest corner
        if x_pos < x_min & y_pos > y_max:  # Top-left corner
            distance_to_wall = np.sqrt((x_pos - x_min)**2 + (y_pos - y_max)**2)
        elif x_pos > x_max & y_pos > y_max:  # Top-right corner
            distance_to_wall = np.sqrt((x_pos - x_max)**2 + (y_pos - y_max)**2)
        elif x_pos < x_min & y_pos < y_min:  # Bottom-left corner
            distance_to_wall = np.sqrt((x_pos - x_min)**2 + (y_pos - y_min)**2)
        elif x_pos > x_max & y_pos < y_min:  # Bottom-right corner
            distance_to_wall = np.sqrt((x_pos - x_max)**2 + (y_pos - y_min)**2)
        
        # If the particle is horizontally aligned with the rectangle, calculate vertical distance
        elif x_min <= x_pos <= x_max and y_pos > y_max:  # Above the rectangle
            distance_to_wall = y_pos - y_max
        elif x_min <= x_pos <= x_max and y_pos < y_min:  # Below the rectangle
            distance_to_wall = y_min - y_pos
        
        # If the particle is vertically aligned with the rectangle, calculate horizontal distance
        elif y_min <= y_pos <= y_max and x_pos < x_min:  # Left of the rectangle
            distance_to_wall = x_min - x_pos
        elif y_min <= y_pos <= y_max and x_pos > x_max:  # Right of the rectangle
            distance_to_wall = x_pos - x_max
        
        # If the particle is inside the rectangle, distance is zero
        else:
            distance_to_wall = -1
        return distance_to_wall
    
    def ajust_initial_positions(self, positions, L):
        """Adjusts the initial positions so they are no longer within the boundary.

        Args:
            positions (numpy array): array of positions
            L (float): size of box the particles can spawn in

        Returns:
            positions        
        """
        # global L
        for i in range(len(positions)):
            # Check if the particle is too close to the wall
            while self.rectangle_wall_filter(positions[i][0], positions[i][1]) <= self.wall_distance:
                # Regenerate position until it is far enough from the wall
                positions[i] = np.random.uniform(0, L, size=(1, 2))
        return positions
        