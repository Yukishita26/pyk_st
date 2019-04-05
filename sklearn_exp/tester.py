import numpy as np
import matplotlib.pyplot as plt

def test():
    n = 50; N = 1000

    x = np.linspace(-3, 3, n)
    X = np.linspace(-3, 3, N)

    pix = np.pi * x
    y = np.sin(pix) / pix + 0.1 * x + 0.1 * np.random.randn(n)

    x = x.reshape(-1, 1)
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    plt.scatter(x,y)
    #plt.show()
