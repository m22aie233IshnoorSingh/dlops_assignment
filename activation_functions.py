import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

print("ReLU Results:", [relu(x) for x in random_values])
print("Leaky ReLU Results:", [leaky_relu(x) for x in random_values])
print("Tanh Results:", [tanh(x) for x in random_values])
