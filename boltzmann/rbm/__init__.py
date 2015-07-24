import numpy as np
from utils import sigmoid


class RBM(object):

    def __init__(self, num_visible, num_hidden, learning_rate=0.1, sparsity_target=None, regulatization_param=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # self.w = np.random.uniform(
        #     low=-1, high=1, size=(num_visible, num_hidden)
        # )
        self.w = np.random.normal(loc=0.0, scale=1., size=(num_visible, num_hidden))
        self.b_vis = np.zeros(num_visible)
        self.b_hid = np.zeros(num_hidden)
        self.learning_rate = learning_rate
        self.sparsity_target = sparsity_target
        self.regulatization_param = regulatization_param

    def sample_hvh(self, h):
        v_probs, v_states = self.sample_v_given_h(h)
        h_probs, h_states = self.sample_h_given_v(v_states)
        return (v_probs, v_states, h_probs, h_states)

    def train(self, data, epochs):
        num_examples = float(data.shape[0])
        for e in xrange(epochs):
            h0_probs, h0_states = self.sample_h_given_v(data)
            v1_probs, v1_states, h1_probs, h1_states = self.sample_hvh(h0_states)

            self.w += self.learning_rate * ((np.dot(data.T, h0_states) - np.dot(v1_states.T, h1_probs)) / num_examples)

            self.b_vis += self.learning_rate * np.mean((data - v1_states), axis=0)
            self.b_hid += self.learning_rate * (np.mean((h0_states - h1_states), axis=0))

            if self.sparsity_target:
                h_mean = np.mean(h0_states, axis=0)
                print h_mean
                sp = (self.sparsity_target - h_mean)
                self.w += self.regulatization_param * sp
                self.b_hid += self.regulatization_param * sp
            print 'epoch', e, 'reconstruction error', self.err(data, v1_probs)

    def err(self, data, recon):
        return np.sum((data - recon) ** 2) / data.shape[0]