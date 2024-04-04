import numpy as np
#in this fix we try to use numpy array for better handling of values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
sigmoid_results = sigmoid(random_values)

print("Sigmoid Results:", sigmoid_results)
