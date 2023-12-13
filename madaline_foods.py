import streamlit as st
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def train_neural_network(inputs, target_output, learning_rate, epochs):
    # Neural network parameters
    input_layer_size = inputs.shape[1]
    hidden_layer_size = 4
    output_layer_size = 1

    # Initialize weights and biases
    weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
    bias_hidden = np.zeros((1, hidden_layer_size))
    weights_hidden_output = np.random.rand(
        hidden_layer_size, output_layer_size)
    bias_output = np.zeros((1, output_layer_size))

    # Training the neural network
    for epoch in range(epochs):
        # Forward pass
        hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(
            hidden_output, weights_hidden_output) + bias_output
        final_output = sigmoid(final_input)

        # Backpropagation
        error = target_output - final_output
        output_delta = error * sigmoid_derivative(final_output)

        hidden_layer_error = output_delta.dot(weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * \
            sigmoid_derivative(hidden_output)

        # Update weights and biases
        weights_hidden_output += hidden_output.T.dot(
            output_delta) * learning_rate
        bias_output += np.sum(output_delta, axis=0,
                              keepdims=True) * learning_rate

        weights_input_hidden += inputs.T.dot(
            hidden_layer_delta) * learning_rate
        bias_hidden += np.sum(hidden_layer_delta, axis=0,
                              keepdims=True) * learning_rate

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output


def test_neural_network(test_input, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_layer_test = sigmoid(
        np.dot(test_input, weights_input_hidden) + bias_hidden)
    output_layer_test = sigmoid(
        np.dot(hidden_layer_test, weights_hidden_output) + bias_output)
    return output_layer_test


# Streamlit UI
st.title("FOOD ALLOTMENT SYSTEM USING MADALINE NEURAL NETWORK")

# Sidebar for user input
st.sidebar.header("Neural Network Training Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 100, 10000, 1000)

# Input data (example values)
inputs = np.array([
    [0.7, 0.5, 0.2],
    [0.2, 0.3, 0.8],
    [0.9, 0.7, 0.4]
])

# Target output (example values)
target_output = np.array([
    [0.8],
    [0.3],
    [0.9]
])

# Train the neural network
weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = train_neural_network(
    inputs, target_output, learning_rate, epochs)

# Testing the neural network
st.header("Test the Neural Network")

# User input for testing
test_input_1 = st.number_input(
    "Customer Satisfaction", 0.0, 1.0, 0.6, step=0.01)
test_input_2 = st.number_input(
    "Ingredient Availability", 0.0, 1.0, 0.6, step=0.01)
test_input_3 = st.number_input("Chef Workload", 0.0, 1.0, 0.7, step=0.01)

# Test the neural network with user input
test_input = np.array([[test_input_1, test_input_2, test_input_3]])
predicted_output = test_neural_network(
    test_input, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

# Display results
st.subheader("Test Result:")
st.write("Predicted Food Allotment:", predicted_output[0, 0])

# Optionally, you can display additional information or charts as needed
