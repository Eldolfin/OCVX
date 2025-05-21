import numpy as np
from random import random


def gradient_descent(f, nb_epochs: int, learning_rate=0.1):
    params = random()
    step = 0.1
    for i in range(nb_epochs):
        samples = [f(i * step) for i in range(-10, 10, 1)]
        params_grad = np.gradient(samples) / step
        indice = min(round(params / step), len(params_grad) - 1)
        params = params - learning_rate * params_grad[indice]
    return params


def main():
    learning_rate = 0.02
    epochs = 70
    expected = 2

    f = lambda x: (x - 1) ** 2 + 1

    res = gradient_descent(f, epochs, learning_rate=learning_rate)
    mse = (res - expected) ** 2

    print("res:", res)
    print("mse:", mse)


if __name__ == "__main__":
    main()
