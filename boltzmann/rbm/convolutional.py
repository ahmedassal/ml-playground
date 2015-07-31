from boltzmann.rbm import RBM
from utils import get_mnist, sigmoid, binarize
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


class ConvolutionalRBM(RBM):

    """
    Uses gaussian visible units,
    a single-layer RBM, so that pooling
    is kinda redundant (pooling groups are
    equal to 1), but implemented anyway.

    Remember, num_hidden == num_filters
    """
    sigma = 1.

    def __init__(self, img_size, feature_map_size, num_feature_maps,
                 learning_rate=0.01, sparsity_target=None,
                 regulatization_param=None):
        self.learning_rate = learning_rate
        self.sparsity_target = sparsity_target
        self.regulatization_param = regulatization_param
        self.fm_height, self.fm_width = feature_map_size
        self.img_height, self.img_width = img_size
        self.num_fm = num_feature_maps
        self.num_channels = 1
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(
            self.num_channels, self.num_fm,
            self.fm_height, self.fm_width
        ))
        self.b_vis = 0
        self.b_hid = np.zeros(self.num_fm)

    def propup(self, v):
        # stacking 2d convolutions here along depth dimension
        # https://github.com/lmjohns3/py-rbm/blob/master/lmj/rbm.py seems
        # to use 1-d convolutions, and I'm not sure is that's ok
        # not going to escape a couple of loops though

        # using theano's conventions:
        # h is 4d matrix (num_examples, num_feature_maps,
        # feature_map_height, feature_map_width)
        # one feature map kinda corresponds to one hidden unit
        # by the same convention, v is 4d matrix too: (num_examples,
        # num_images per example (1, or 3 for RGB), image_height,
        # image_widht)
        # the same format is for weights: (number of feature maps for visible
        # layer (1 or 3), number of feature maps for hidden layer,
        # filter height, filter width)

        num_examples = v.shape[0]
        activations = np.zeros(
            (
                num_examples,
                self.num_fm,
                self.img_height - self.fm_height + 1,
                self.img_width - self.fm_width + 1
            )
        )
        for i in xrange(num_examples):
            for j in xrange(self.num_fm):
                activations[i, j, :, :] = convolve2d(v[i, 0, :, :], self.w[0, j, ::-1, ::-1], mode='valid')
        return sigmoid(activations + self.b_hid[None, :, None, None])

    def propdown(self, h):
        # here's a curious thing happens.
        # since RBM's weights are symmetric, we have to perform
        # convolution backwards and therefore to have same-sized feature maps
        # therefore 'full' convolution mode (with padding) is applied
        num_examples = h.shape[0]
        v = np.zeros(
            (num_examples, self.num_channels, self.img_height, self.img_width)
        )

        for i in xrange(num_examples):
            for j in xrange(self.num_fm):
                v[i, 0, :, :] += convolve2d(h[i, j, :, :], self.w[0, j, :, :], mode='full')
        return sigmoid(v + self.b_vis)

    def sample_hidden(self, activations):
        return np.random.binomial(n=1, p=activations)

    def sample_visible(self, arr):
        return np.random.binomial(n=1, p=arr)

    def sample_h_given_v(self, v):
        probs = self.propup(v)
        states = self.sample_hidden(probs)
        return probs, states

    def sample_v_given_h(self, h):
        probs = self.propdown(h)
        states = self.sample_visible(probs)
        return probs, states

    def train(self, data, epochs, visualize=False):
        if visualize:
            plt.ion()
        num_examples = float(data.shape[0])
        for e in xrange(epochs):
            h0_probs, h0_states = self.sample_h_given_v(data)
            v1_probs, v1_states, h1_probs, h1_states = self.sample_hvh(h0_states)
            # for i in xrange(5):
            #     plt.subplot(1, 5, i)
            #     img = v1_states[i, 0, ...]
            #     plt.xticks(())
            #     plt.yticks(())
            #     plt.imshow(img, cmap='gray', interpolation='none')
            # # plt.draw()

            delta_w = np.zeros_like(self.w)

            for j in xrange(self.num_fm):
                # pos = []
                # neg = []
                diffs = []
                for i in xrange(int(num_examples)):
                    pos = convolve2d(data[i, 0, :, :], (h0_probs[i, j, ::-1, ::-1]), mode='valid')
                    neg = convolve2d(v1_states[i, 0, :, :], (h1_probs[i, j, ::-1, ::-1]), mode='valid')
                    diffs.append(pos - neg)
                delta_w[0, j, :, :] = np.mean(diffs, axis=0)

            delta_b_vis = np.sum(data - v1_states) / num_examples
            delta_b_hid = (h0_states - h1_states).sum(axis=-1).sum(axis=-1).mean(axis=0)

            self.w += self.learning_rate * delta_w
            self.b_vis += self.learning_rate * delta_b_vis
            self.b_hid += self.learning_rate * delta_b_hid

            if self.sparsity_target:
                h_mean = np.mean(h0_probs, axis=(0, 2, 3))
                print h_mean
                sp = (self.sparsity_target - h_mean)
                self.b_hid += (self.regulatization_param * sp)
            print 'epoch', e, 'reconstruction error', self.err(data, v1_states)
            if visualize:
                for i in xrange(h):
                    plt.subplot(x, y, i)
                    img = self.w[0, i, :, :]
                    plt.xticks(())
                    plt.yticks(())
                    plt.imshow(img, cmap='gray', interpolation='none')
                plt.draw()
        if visualize:
            plt.ioff()


if __name__ == '__main__':
    m = 5
    dim = 28
    data, target = get_mnist(random=True, num=m)
    data = binarize(data, 0., 1.)

    data = data.reshape(m, 1, dim, dim)

    h = 5
    x = 1
    y = 5

    rbm = ConvolutionalRBM(
        img_size=(dim, dim),
        feature_map_size=(13, 13),
        num_feature_maps=h,
        learning_rate=0.001,
        sparsity_target=0.1,
        regulatization_param=10
    )
    rbm.train(data, epochs=100, visualize=True)

    for i in xrange(h):
        plt.subplot(x, y, i)
        img = rbm.w[0, i, :, :]
        plt.xticks(())
        plt.yticks(())
        plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()
