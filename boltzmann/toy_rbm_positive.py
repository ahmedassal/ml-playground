import numpy as np


class PositiveToyRBM(object):
    """
    This is an example of RBM using positive phase only.
    Made for demonstration only. Notice also this RBM is
    deterministic (it doesn't use stochastic activation
    function) and the learning algorithm I'm trying to
    implement is not contrastive divergence but the
    original BM-learning algorithm by Ackley, Hinton and
    Sejnowski.

    Details:

     * Ackley, David H.; Hinton, Geoffrey E.; Sejnowski, Terrence J. (1985).
       "A Learning Algorithm for Boltzmann Machines".
     * rocknrollnerd.github.io/ml/2015/07/19/general-boltzmann-machines.html
    """

    def __init__(self, num_visible, num_hidden, w=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        if w is None:
            self.w = np.zeros((num_visible, num_hidden))
        else:
            self.w = np.float32(w)

    def threshold(self, arr, tr=0):
        if isinstance(arr, np.ndarray):
            arr[arr >= tr] = 1
            arr[arr < tr] = -1
            return arr
        else:
            return 1 if arr >= 0 else -1

    def hebbian(self, visible, hidden):
        # for each pair of units determine if they are both on
        return np.dot(visible.reshape(visible.shape[0], 1),
                      hidden.reshape(hidden.shape[0], 1))

    def pp(self, arr):
        # pretty print
        return list([list(i) for i in arr])

    def try_reconstruct(self, data):
        h = self.threshold(np.dot(data, self.w))
        recon = self.threshold(np.dot(h, self.w.T))
        return np.sum(data - recon) == 0

    def train(self, data, epochs=10):
        data = np.array(data)
        for e in xrange(epochs):
            delta_w = []
            for example in data:
                h = self.threshold(np.dot(example, self.w))
                delta_w.append(self.hebbian(example, h))
            # average
            delta_w = np.mean(delta_w, axis=0)
            self.w += delta_w
            result = self.try_reconstruct(data)
            print 'epoch', e, 'delta w =', self.pp(delta_w), 'new weights =', self.pp(self.w), 'reconstruction ok?', result
            if result:
                break

if __name__ == '__main__':
    rbm = PositiveToyRBM(3, 1, w=[[0], [0], [0]])
    rbm.train([[1, -1, 1], [-1, 1, -1]])
    print "This won't ever end up well, because we're only using positive phase :-("