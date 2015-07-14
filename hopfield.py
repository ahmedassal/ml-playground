import numpy as np
import matplotlib.pyplot as plt

from utils import get_mnist


class HopfieldNet(object):

    def __init__(self, num_units):
        self.learning_rate = 1
        self.num_units = num_units
        self.w = np.zeros((num_units, num_units))
        self.b = np.zeros(num_units)
        self.threshold = 0

    def store(self, data):
        for j, item in enumerate(data):
            item = item.reshape(item.shape[0], 1)
            activations = np.dot(item, item.T)
            np.fill_diagonal(activations, 0)
            self.w += activations
            self.b += item.ravel()

    def get_energy(self, _data):
        # first let again compute quadratic activations
        data = _data.reshape(_data.shape[0], 1)
        activations = np.float32(np.dot(data, data.T))
        np.fill_diagonal(activations, 0)
        # then multiply each activation by weight an
        activations *= self.w
        weight_term = np.sum(activations) / 2  # divide by 2, because we've counted them twice
        bias_term = np.dot(self.b, data)[0]
        return -bias_term - weight_term

    def restore(self, _data):
        data = np.copy(_data)
        idx = range(len(data))
        # TODO: how many loops required?
        for i in xrange(10):
            for _ in xrange(len(data)):
                j = np.random.choice(idx)
                inputs = np.sum(data * self.w[j])
                if inputs > self.threshold:
                    data[j] = 1
                else:
                    data[j] = -1
        return data


def binarize(_arr):
    arr = np.float32(_arr)
    arr[arr <= 0] = -1
    arr[arr > 0] = 1
    return arr


if __name__ == '__main__':
    # prepare initial data
    data1, target1 = get_mnist(0, 1)
    data1 = data1.ravel()
    data = np.vstack([data1])
    data = binarize(data)

    net = HopfieldNet(28 * 28)
    net.store(data)

    # get noisy example
    example = binarize(data1)
    noise_idx = np.random.choice(range(28 * 28), 250)
    example[noise_idx] *= -1

    plt.imshow(example.reshape(28, 28), cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    restored = net.restore(example)

    plt.imshow(restored.reshape(28, 28), cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()
