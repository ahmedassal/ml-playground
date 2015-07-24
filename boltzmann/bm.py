import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from utils import get_mnist


class BM(object):

    def __init__(self, num_visible, num_hidden, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.w = np.random.uniform(
            low=-1, high=1, size=(
                num_visible + num_hidden,
                num_visible + num_hidden
            )
        )
        self.w = self.symmetrize(self.w)
        self.b = np.zeros(num_visible + num_hidden)
        self.temperature = 1.

    def symmetrize(self, arr):
        # set main diagonal to zero and mirror the matrix
        arr = np.tril(arr) + np.tril(arr, -1).T
        np.fill_diagonal(arr, 0)
        return arr

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def train(self, data, epochs, visualize=False):
        if visualize:
            plt.ion()
        for e in xrange(epochs):
            print e
            positive_stats = self.positive_phase(data)
            negative_stats = self.negative_phase(data.shape[0] * 2)
            self.update(positive_stats, negative_stats)
            if visualize:
                self.draw_weights()
                plt.draw()
        if visualize:
            plt.ioff()

    def positive_phase(self, external_data):
        # set hidden units to random values
        # and concat to external data
        w_stats = []
        b_stats = []
        for example in external_data:
            # prepare data
            hidden = np.random.choice([1., 0.], size=self.num_hidden)
            state = np.hstack([hidden, example])

            # now update hidden units until thermal equilibrium
            # a curious thing is I still don't know when it happens...
            # so a fixed number of epochs for now
            choices = np.random.choice(range(self.num_hidden), 100)
            for i in choices:
                state = self.update_neuron(state, i)

            # then collect stats
            w_stats_example, b_stats_example = self.collect_stats(state)
            w_stats.append(w_stats_example)
            b_stats.append(b_stats_example)

        # average stats across training examples
        w_stats = np.mean(w_stats, axis=0)
        b_stats = np.mean(b_stats, axis=0)
        return w_stats, b_stats

    def negative_phase(self, epochs):
        w_stats = []
        b_stats = []
        for e in xrange(epochs):
            # set random state
            state = np.random.choice([1., 0.], size=self.num_hidden + self.num_visible)

            # again, wait for equilibrium
            # but now update all neurons, not just hidden
            choices = np.random.choice(range(self.num_hidden + self.num_visible), 1000)
            for i in choices:
                state = self.update_neuron(state, i)

            # and collect stats
            w_stats_example, b_stats_example = self.collect_stats(state)
            w_stats.append(w_stats_example)
            b_stats.append(b_stats_example)

        # and average
        w_stats = np.mean(w_stats, axis=0)
        b_stats = np.mean(b_stats, axis=0)
        return w_stats, b_stats

    def update(self, positive_stats, negative_stats):
        w_pos, b_pos = positive_stats
        w_neg, b_neg = negative_stats

        self.b += self.learning_rate * (b_pos - b_neg)
        self.w += self.learning_rate * (w_pos - w_neg)

    def update_neuron(self, state, i):
        dE = np.dot(self.w[i], state) + self.b[i]
        prob = self.sigmoid(dE / self.temperature)
        if prob >= np.random.rand():
            state[i] = 1.
        else:
            state[i] = 0.
        return state

    def collect_stats(self, state):
        # collecting corellations between pairs of neurons
        b_stats = state.copy()
        b_stats = (b_stats - 0.5) * 2.
        _state = state.reshape(state.shape[0], 1)
        w_stats = np.dot(_state - 0.5, _state.T - 0.5) * 4.
        np.fill_diagonal(w_stats, 0)
        return w_stats, b_stats

    def draw_weights(self):
        hw = net.w[:self.num_hidden, :]
        for i in xrange(self.num_hidden):
            plt.subplot(1, self.num_hidden, i)
            img = hw[i][self.num_hidden:].reshape(dim, dim)
            plt.xticks(())
            plt.yticks(())
            plt.imshow(img, cmap='gray', interpolation='none')


def binarize(_arr):
    arr = np.float32(_arr)
    arr[arr <= 0] = 0.
    arr[arr > 0] = 1.
    return arr


def get_preprocessed_mnist(start, end):
    data, target = get_mnist(start, end)
    return binarize(np.vstack([
        resize(item.reshape(28, 28), (dim, dim)).ravel()
        for item in data
    ]))


if __name__ == '__main__':
    dim = 16  # resize dimention
    h = 4  # number of hidden units

    # uncomment selecting random set of MNIST digits
    # data, target = get_mnist(random=True, num=200)
    # data = binarize(np.vstack([
    #     resize(item.reshape(28, 28), (dim, dim)).ravel()
    #     for item in data
    # ]))

    # just a subset of 3 different digit classes
    data = np.vstack([
        get_preprocessed_mnist(10000, 10010),
        get_preprocessed_mnist(30000, 30010),
        get_preprocessed_mnist(40000, 40010),
    ])

    net = BM(dim * dim, h, 0.5)
    net.train(data, epochs=100, visualize=False)
    net.draw_weights()
    plt.show()
