import numpy as np
import matplotlib.pyplot as plt

class GradientDescentVisualizer:
    """
        Initialize the visualizer with the necessary parameters.

        :param function: The function to optimize (minimize).
        :param derivative: The derivative of the function.
        :param start_x: Starting x-coordinate for gradient descent.
        :param learning_rate: Step size for each iteration.
        :param steps: Maximum number of iterations for gradient descent.
        :param epsilon: Threshold for convergence.
        :param x_range: Range for x-axis (tuple of min and max x values).
    """
    def __init__(self, function, derivative, start_x=45, learning_rate=0.01, steps=250, epsilon = 1e-5, x_range=(-50, 50)):
        self.function = function
        self.derivative = derivative
        self.start_x = start_x
        self.learning_rate = learning_rate
        self.steps = steps
        self.epsilon = epsilon
        self.x_range = x_range
    
    def run(self):
        x_values = np.arange(self.x_range[0], self.x_range[1], 0.1)
        y_values = self.function(x_values)

        # starting postion for gradent descent
        current_x = self.start_x
        current_y = self.function(current_x)

        #plot settings
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_values, y_values, label=f"{self.function.__name__}(x)", color="blue")
        ax.set_title("Gradient Descent Visualization with Epsilon Convergence")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        for step in range(self.steps):
            # updating gradent descent
            old_x, old_y = current_x, current_y
            current_x -= self.learning_rate * self.derivative(current_x)
            current_y = self.function(current_x)

            # Plot updates
            ax.plot(x_values, y_values, color="blue")  # Main function
            ax.scatter(current_x, current_y, color="red", s=50, zorder=5, label="Current Position" if step == 0 else "")
            ax.legend(loc="upper left")

            # Display step details on the plot
            ax.text(0.02, 0.95, f"Step: {step+1}\nLearning rate: {self.learning_rate}\nPosition: ({current_x:.2f}, {current_y:.2f})\nChange: {abs(current_y - old_y):.6f}",
                    transform=ax.transAxes, verticalalignment="top", color="green", bbox=dict(facecolor="white", alpha=0.8))

            # Check for convergence based on epsilon
            if abs(current_y - old_y) < self.epsilon:
                print(f"Converged in {step+1} steps with change {abs(current_y - old_y):.6f}")
                break

            plt.pause(0.01)
            ax.cla()  # Clear axis for next plot update

        plt.ioff()
        plt.show()


def quadratic_function(x):
    return x**2 + 3*x + 5

def quadratic_derivative(x):
    return 2*x + 3


# Instantiate and run the visualizer
visualizer = GradientDescentVisualizer(
    function=quadratic_function,
    derivative=quadratic_derivative,
    learning_rate=0.01,
    steps=10000,
    epsilon=1e-5
)
visualizer.run()