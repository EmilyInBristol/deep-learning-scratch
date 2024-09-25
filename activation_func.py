import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Tanh function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Gradient Descent: Train using a simple linear neuron
def gradient_descent(activation_func, activation_derivative, lr=0.1, epochs=100):
    # Initialize weight and bias
    weight = np.random.randn()
    bias = np.random.randn()

    # Simple dataset: Single input and single target output
    X = np.array([0.5])  # Input
    y_true = np.array([1])  # Target output

    # Track loss for plotting
    losses = []

    for epoch in range(epochs):
        # Forward pass: Linear combination + activation
        z = weight * X + bias
        y_pred = activation_func(z)

        # Compute the loss (Mean Squared Error)
        loss = 0.5 * (y_pred - y_true) ** 2
        losses.append(loss)

        # Backward pass: Compute gradient of loss w.r.t weight and bias
        dL_dy_pred = y_pred - y_true  # Derivative of loss w.r.t. output
        dy_pred_dz = activation_derivative(z)  # Derivative of output w.r.t. z
        dz_dw = X  # Derivative of z w.r.t. weight
        dz_db = 1  # Derivative of z w.r.t. bias

        # Chain rule to get the gradients for weight and bias
        dL_dw = dL_dy_pred * dy_pred_dz * dz_dw
        dL_db = dL_dy_pred * dy_pred_dz * dz_db

        # Update weight and bias
        weight -= lr * dL_dw
        bias -= lr * dL_db

    return losses

# Hyperparameters
learning_rate = 0.1
epochs = 100

# Train using Sigmoid
losses_sigmoid = gradient_descent(sigmoid, sigmoid_derivative, lr=learning_rate, epochs=epochs)

# Train using Tanh
losses_tanh = gradient_descent(tanh, tanh_derivative, lr=learning_rate, epochs=epochs)

# Train using ReLU
losses_relu = gradient_descent(relu, relu_derivative, lr=learning_rate, epochs=epochs)

# Plot the loss over epochs for sigmoid, tanh, and relu
plt.plot(losses_sigmoid, label='Sigmoid', color='blue')
plt.plot(losses_tanh, label='Tanh', color='red')
plt.plot(losses_relu, label='ReLU', color='green')
plt.title('Gradient Descent: Sigmoid vs Tanh vs ReLU Activation Function')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
