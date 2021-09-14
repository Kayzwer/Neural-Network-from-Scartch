import numpy as np

def gradient_descent(x, y, w, b, iteration):
    n = len(x)
    learning_rate = 1e-3

    for i in range(iteration):
        y_hat = w * x + b
        loss = sum(y - y_hat) ** 2
        dw = -2 * sum(x * (y - y_hat))
        db = -2 * sum(y - y_hat)
        w -= learning_rate * dw
        b -= learning_rate * db
        print(f"Epoch: {i + 1}, Loss: {loss}, Weight: {w}, Bias: {b} \n")


if __name__ == '__main__':

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.5, 1, 1.5, 2, 2.5])
    w = 0
    b = 0

    gradient_descent(x, y, w, b, 10000)
