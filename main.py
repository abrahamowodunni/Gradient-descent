import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some random data
np.random.seed(42)  # For reproducibility
x = np.random.randn(100, 1)  # 100 samples
y = 2 * x + np.random.rand(100, 1) * 0.5  # Linear relation with noise

# Parameters
weight = 0.0
bias = 0.0
# Hyperparameters
lr = 0.1
epsilon = 1e-4  # Small value to avoid tiny updates

# Create a function to compute MSE
def compute_mse(x, y, weight, bias):
    N = x.shape[0]
    yhat = weight * x + bias
    return np.sum((y - yhat) ** 2) / N

# Create a function to calculate gradients
def grad_descent(x, y, weight, bias, lr, epsilon):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (weight * xi + bias))
        dldb += -2 * (yi - (weight * xi + bias))
    
    # Update parameters with epsilon adjustment
    weight -= lr * (1/N) * dldw if abs(dldw) > epsilon else 0
    bias -= lr * (1/N) * dldb if abs(dldb) > epsilon else 0

    return weight, bias

# Create a grid of weights and biases for the MSE surface
weight_range = np.linspace(-4, 4, 100)
bias_range = np.linspace(-4, 4, 100)
W, B = np.meshgrid(weight_range, bias_range)
Z = np.array([[compute_mse(x, y, w, b) for w in weight_range] for b in bias_range])

# Create a 3D plot for the MSE surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the MSE surface
surf = ax.plot_surface(W, B, Z, alpha=0.5, cmap='viridis', edgecolor='none')
ax.set_xlabel('Weight', fontsize=12)
ax.set_ylabel('Bias', fontsize=12)
ax.set_zlabel('MSE', fontsize=12)
ax.set_title('MSE Surface with Gradient Descent Visualization', fontsize=14, fontweight='bold')

# Set the viewing angle
ax.view_init(elev=30, azim=210)

# Plotting the gradient descent process
history = []  # Store the history of weights and biases for plotting
for epoch in range(150):  # Reduced number of epochs for faster animation
    weight, bias = grad_descent(x, y, weight, bias, lr, epsilon)
    
    # Update the MSE after the parameters are updated
    current_mse = compute_mse(x, y, weight, bias)
    
    # Store the history for later visualization
    history.append([weight, bias, current_mse])  # Append as a list instead of a tuple

    # Plot the current position of weight and bias
    ax.scatter(weight, bias, current_mse, color='red', s=50, alpha=0.7)  # Point moving down the surface
    plt.pause(0.01)  # Decrease pause duration for faster movement

# Convert history to a NumPy array
history = np.array(history)

# Plot the trajectory of the descent
ax.plot(history[:, 0], history[:, 1], history[:, 2], color='blue', linewidth=2, alpha=0.7, label='Gradient Descent Path')
ax.legend()

# Print final results
final_loss = compute_mse(x, y, weight, bias)
print(f'Final weight: {weight}, Final bias: {bias}, Final loss: {final_loss}')

# Annotate the final result in the plot
ax.text(weight, bias, final_loss, f'Final Weight: {weight:.2f}\nFinal Bias: {bias:.2f}\nFinal Loss: {final_loss:.2f}',
        color='black', fontsize=10, ha='center')

plt.show()
