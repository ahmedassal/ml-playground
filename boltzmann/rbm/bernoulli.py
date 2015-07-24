from boltzmann.rbm import RBM
from utils import get_mnist, binarize, sigmoid
import matplotlib.pyplot as plt
import numpy as np


class BernoulliRBM(RBM):
    """
    Simple RBM with binary units (sparsity enabled!)
    """
    def propup(self, v):
        # computes probability of hidden units given visible
        return sigmoid(np.dot(v, self.w) + self.b_hid)

    def propdown(self, h):
        # computes probability of visible units given hidden
        return sigmoid(np.dot(h, self.w.T) + self.b_vis)

    def sample(self, arr):
        return np.random.binomial(n=1, p=arr)

    def sample_h_given_v(self, v):
        probs = self.propup(v)
        states = self.sample(probs)
        return probs, states

    def sample_v_given_h(self, h):
        probs = self.propdown(h)
        states = self.sample(probs)
        return probs, states

if __name__ == '__main__':
    data, target = get_mnist(random=True, num=1000)
    data = binarize(data, 0., 1.)

    dim = 28
    h = 100
    x = 10
    y = 10

    rbm = BernoulliRBM(
        dim * dim, h, learning_rate=0.1,
        sparsity_target=0.05,
        regulatization_param=0.01
        )
    rbm.train(data, epochs=1000)

    for i in xrange(h):
        plt.subplot(x, y, i)
        img = rbm.w[:, i].reshape(dim, dim)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()
