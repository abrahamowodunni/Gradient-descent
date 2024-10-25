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
        fig = plt.figure(figsize=(12, 6))
        
        # Prepare the grid for the function surface
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.z_function(X, Y)

        # Create a 3D surface plot
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.set_title("3D Surface Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis (f(x,y))")

        # Create a contour plot
        ax_contour = fig.add_subplot(122)
        ax_contour.contour(X, Y, Z, levels=20, cmap='viridis')
        ax_contour.set_title("Contour Plot")
        ax_contour.set_xlabel("X-axis")
        ax_contour.set_ylabel("Y-axis")
        
        # Plot the start position
        ax.scatter(self.initial_position[0], self.initial_position[1], self.initial_position[2], color='red', s=50, label='Start')
        ax_contour.scatter(self.initial_position[0], self.initial_position[1], color='red', s=50)

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

            # Update the surface plot
            ax.cla()
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
            ax.scatter(self.initial_position[0], self.initial_position[1], self.initial_position[2], color='red', s=50)
            ax.scatter(x_new, y_new, z_new, color='green', s=50, label='Current Position')
            ax.set_title("3D Surface Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis (f(x,y))")

            # Update the contour plot
            ax_contour.cla()
            ax_contour.contour(X, Y, Z, levels=20, cmap='viridis')
            ax_contour.scatter(x_new, y_new, color='green', s=50, label='Current Position')
            ax_contour.scatter(self.initial_position[0], self.initial_position[1], color='red', s=50)
            ax_contour.set_title("Contour Plot")
            ax_contour.set_xlabel("X-axis")
            ax_contour.set_ylabel("Y-axis")
            
            # Show the current frame
            plt.pause(0.1)

        plt.show()

# Define a simple paraboloid function and its gradient
def z_function(x, y):
    return x**2 + y**2  # A simple quadratic function

def calculate_gradient(x, y):
    return (2 * x, 2 * y)  # Gradient of the quadratic function

# Instantiate the optimizer
optimizer = GradientDescentOptimizer(z_function, calculate_gradient, learning_rate=0.09, max_iterations=100)
optimizer.set_initial_position(1.5, 1.5)

# Perform optimization and visualize the results
optimizer.optimize()
