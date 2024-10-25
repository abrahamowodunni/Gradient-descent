import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GradientDescentOptimizer:
    def __init__(self, z_function, gradient_function, learning_rate=0.01, max_iterations=1000, epsilon=1e-6):
        self.z_function = z_function
        self.gradient_function = gradient_function
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.initial_position = None
        self.current_position = None
        
    def set_initial_position(self, x, y):
        """Set the initial position for the gradient descent."""
        self.initial_position = (x, y, self.z_function(x, y))
        self.current_position = self.initial_position
    
    def optimize(self):
        """Perform the gradient descent optimization."""
        x_history = []
        y_history = []
        z_history = []
        
        # Create a figure for visualization
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Prepare the grid for the function surface
        x_range = np.arange(-1, 1, 0.05)
        y_range = np.arange(-1, 1, 0.05)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.z_function(X, Y)

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

        # Plot the start position
        ax.scatter(self.initial_position[0], self.initial_position[1], self.initial_position[2], color='red', s=50, label='Start')

        for _ in range(self.max_iterations):
            x_derivative, y_derivative = self.gradient_function(self.current_position[0], self.current_position[1])
            x_new = self.current_position[0] - self.learning_rate * x_derivative
            y_new = self.current_position[1] - self.learning_rate * y_derivative
            z_new = self.z_function(x_new, y_new)

            # Store history for visualization
            x_history.append(x_new)
            y_history.append(y_new)
            z_history.append(z_new)

            # Check for convergence
            if np.abs(z_new - self.current_position[2]) < self.epsilon:
                break

            # Update current position
            self.current_position = (x_new, y_new, z_new)

            # Clear the plot for the next frame
            ax.cla()

            # Replot the surface
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

            # Plot the updated current position
            ax.scatter(x_new, y_new, z_new, color='green', s=50, label='Current Position')

            # Plot the start position again
            ax.scatter(self.initial_position[0], self.initial_position[1], self.initial_position[2], color='red', s=50)

            # Update plot aesthetics
            ax.set_title("Gradient Descent Optimization")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis (f(x,y))")
            ax.legend()

            # Show the current frame
            plt.pause(0.01)  # Pause to visualize the movement step by step

        plt.show()

# Define the function and gradient
def z_function(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5

def calculate_gradient(x, y):
    return (5 * np.cos(5 * x) * np.cos(5 * y), -5 * np.sin(5 * x) * np.sin(5 * y))

# Instantiate the optimizer
optimizer = GradientDescentOptimizer(z_function, calculate_gradient, learning_rate=0.01, max_iterations=1000)
optimizer.set_initial_position(0.5, 0.9)

# Perform optimization and visualize the results
optimizer.optimize()
