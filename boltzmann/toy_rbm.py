import numpy as np

from boltzmann.toy_rbm_positive import PositiveToyRBM


class ToyRBM(PositiveToyRBM):
    """
    This is a fixed version of PositiveToyRBM
    with negative phase added, so now it functions
    kinda like a normal Boltzmann Machine. It's still
    deterministic though and doesn't customize enough.
    """

    def collect_negative_stats(self):
        # we don't know in advance how many loops required to reach equilibrium
        stats = []
        for e in xrange(10):
            # initial random state
            visible = self.threshold(np.random.rand(self.num_visible), 0.5)
            hidden = self.threshold(np.random.rand(self.num_hidden), 0.5)
            idx = range(self.num_visible + self.num_hidden)

            # settling for equilibrium
            # again, number of loops is guessed
            for _ in xrange(50):
                i = np.random.choice(idx)  # selecting random neuron
                if i < self.num_visible:  # visible neuron
                    visible[i] = self.threshold(np.sum(self.w[i] * visible[i]))
                else:  # hidden neuron
                    i -= self.num_visible
                    hidden[i] = self.threshold(np.sum(self.w[:, i] * hidden[i]))

            # hopefully done, now make a reconstruction and collect stats
            recon = self.threshold(np.dot(hidden, self.w.T))
            stats.append(self.hebbian(recon, hidden))
        # average
        return np.mean(stats, axis=0)

    def train(self, data, epochs=10, with_negative=False):
        data = np.array(data)
        for e in xrange(epochs):
            delta_w = []
            for example in data:
                h = self.threshold(np.dot(example, self.w))
                delta_w.append(self.hebbian(example, h))
            # average
            delta_w = np.mean(delta_w, axis=0)
            if with_negative:
                delta_w -= self.collect_negative_stats()
            self.w += delta_w
            result = self.try_reconstruct(data)
            print 'epoch', e, 'delta w =', self.pp(delta_w), 'new weights =', self.pp(self.w), 'reconstruction ok?', result
            if result:
                break

if __name__ == '__main__':
    rbm = ToyRBM(3, 1, w=[[0], [0], [0]])
    rbm.train([[1, -1, 1], [-1, 1, -1]], with_negative=True)
