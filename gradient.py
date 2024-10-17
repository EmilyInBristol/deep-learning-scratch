import numpy as np

def f(x):
    return (x-3) ** 2

def gradient(x):
    return 2 * (x-3)

def gradient_descent(lr=0.05, tolerance=0.006, max_iter=100):

    x = 10

    for i in range(max_iter):
        new_x = x - lr * gradient(x)
        if x - new_x < tolerance:
            break
        x = new_x

    return x

#print(gradient_descent())

def standardize(data):
    mean = np.mean(data, axis=0)
    print(mean)
    std = np.std(data, axis=0)
    print(std)
    standardize_data = (data - mean) / std
    return standardize_data

x = np.array([[50, 30], [20, 40], [30, 20], [70, 50]])
res = standardize(x)
print(res.shape)
