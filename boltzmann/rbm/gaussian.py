from boltzmann.rbm import RBM
from utils import get_mnist, sigmoid
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np


class GaussianRBM(RBM):

    """
    Gaussian-Bernoulli RBM (with Gaussian
    visible units, hidden units are still binary)
    """
    sigma = 1.

    def propup(self, v):
        # computes probability of hidden units given visible
        return sigmoid(np.dot(v, self.w) + self.b_hid)

    def propdown(self, h):
        # computes mean for Gaussian distribution
        # for visible units
        return np.dot(h, self.w.T) + self.b_vis

    def sample_hidden(self, arr):
        # binomial
        return np.random.binomial(n=1, p=arr)

    def sample_visible(self, arr):
        # and that's Gaussian
        return self.sigma * np.random.normal(size=arr.shape) + arr

    def sample_h_given_v(self, v):
        probs = self.propup(v)
        states = self.sample_hidden(probs)
        return probs, states

    def sample_v_given_h(self, h):
        probs = self.propdown(h)
        states = self.sample_visible(probs)
        return probs, states

    def train(self, data, epochs, visualize=False):
        # overriding to use states instead of probabilities
        # and to restrict sparsity by adding it to hidden bias only
        if visualize:
            plt.ion()
        num_examples = float(data.shape[0])
        for e in xrange(epochs):
            h0_probs, h0_states = self.sample_h_given_v(data)
            v1_probs, v1_states, h1_probs, h1_states = self.sample_hvh(
                h0_states)

            self.w += self.learning_rate * ((np.dot(data.T, h0_states) - np.dot(v1_states.T, h1_states)) / num_examples)
            self.b_vis += self.learning_rate * (np.sum((data - v1_states), axis=0) / num_examples)
            self.b_hid += self.learning_rate * (np.sum((h0_states - h1_states), axis=0) / num_examples)
            if self.sparsity_target:
                h_mean = np.mean(h0_states, axis=0)
                print h_mean
                sp = (self.sparsity_target - h_mean)
                self.b_hid += self.regulatization_param * sp
            print 'epoch', e, 'reconstruction error', self.err(data, v1_states)
            if visualize:
                for i in xrange(self.num_hidden):
                    plt.subplot(5, 5, i)
                    # img = dewhiten(pca, rbm.w[:, i]).reshape(dim, dim)
                    img = rbm.w[:, i].reshape(28, 28)
                    plt.xticks(())
                    plt.yticks(())
                    plt.imshow(img, cmap='gray', interpolation='none')
                plt.draw()
        if visualize:
            plt.ioff()


def whiten(data, epsilon=0.01):
    # subsctract mean per image
    data = data.T  # reshaping to m x n
    data_mean = data.mean(axis=1)
    data = data - np.tile(data_mean, (data.shape[1], 1)).transpose()
    sigma = data.dot(data.T) / data.shape[1]
    u, s, v = np.linalg.svd(sigma)
    D = np.diag(1. / np.sqrt(np.diag(s) + epsilon))
    D = np.eye(D.shape[0]) * D
    whitened = u.dot(D).dot(u.T).dot(data)
    return whitened.T


if __name__ == '__main__':
    data, target = get_mnist(random=True, num=200)
    data = np.float32(data) / 255.
    data = whiten(data)
    data = preprocessing.scale(data)

    dim = 28
    h = 25
    x = 5
    y = 5

    for i in xrange(100):
        plt.subplot(10, 10, i)
        img = data[i].reshape(28, 28)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()

    rbm = GaussianRBM(
        28 * 28, h, learning_rate=0.1,
        # sparsity_target=0.05,
        # regulatization_param=0.1
    )
    rbm.train(data, epochs=500, visualize=True)

    for i in xrange(h):
        plt.subplot(x, y, i)
        img = rbm.w[:, i].reshape(dim, dim)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()
