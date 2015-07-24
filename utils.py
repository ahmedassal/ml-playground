from sklearn.datasets import fetch_mldata
import numpy as np


def get_mnist(start=None, end=None, random=False, num=None):
    mnist = fetch_mldata('MNIST original', data_home='~/diss/mnist')
    if random is not None and num is not None:
        idx = np.random.choice(range(mnist.data.shape[0]), num)
    elif start is not None and end is not None:
        idx = range(start, end)
    return mnist.data[idx], mnist.target[idx]


def binarize(_arr, min_value=-1, max_value=1):
    arr = np.float32(_arr)
    arr[arr <= 0] = min_value
    arr[arr > 0] = max_value
    return arr


def sigmoid(x):
    return 1.0 / (1. + np.exp(-x))
