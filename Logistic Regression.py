import numpy as np

def gradient_descent(x, y, w, b):
    learning_rate = 1e-2

    for i in range(100000):
        y_hat = 1 / (1 + np.exp(-(w * x + b)))
        loss = -sum(y * np.log(y_hat) + (1 - y) * np.log(1- y_hat))
        dw = sum((y_hat - y) * x)
        db = sum(y_hat - y)
        w -= learning_rate * dw
        b -= learning_rate * db
        print(f"Epoch: {i + 1}, Loss: {loss}, Weight: {w}, Bias: {b} \n")


if __name__ == '__main__':
    x = np.array([-2, -1, 0, 1, 1])
    y = np.array([0, 0, 0, 1, 1])
    w = 0
    b = 0

    gradient_descent(x, y, w, b)
