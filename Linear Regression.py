import numpy as np

def gradient_descent(x, y, w, b, iteration, learning_rate):
    n = len(x)
    for i in range(iteration):
        y_hat = w * x + b
        loss = sum(y - y_hat) ** 2 / n
        dw = -2 / n * sum(x * (y - y_hat))
        db = -2 / n * sum(y - y_hat)
        w -= learning_rate * dw
        b -= learning_rate * db
        print(f"Epoch: {i + 1}, Loss: {loss}, Weight: {w}, Bias: {b} \n")


if __name__ == '__main__':

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    w = 0
    b = 0
    gradient_descent(x, y, w, b, 5000, 0.001)
