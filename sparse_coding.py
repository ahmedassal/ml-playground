from utils import get_patches
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
from sklearn import preprocessing
import theano

theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'


class Trainer(object):

    def __init__(self, X, updates, cost, batch_size, learning_rate=0.1, epochs=1000):
        self.X = X
        self.cost = cost
        self.learning_rate = learning_rate
        self.updates = updates
        self.epochs = epochs
        self.batch_size = batch_size
        self.index = T.iscalar('i')

    def train(self, data, verbose=True):
        n_batches = data.shape[0] / self.batch_size
        # now to be able to make subtensors we need to make our data shared values
        # data = theano.shared(value=data)
        train_function = theano.function(
            inputs=[self.X],
            outputs=self.cost,
            updates=self.updates,
            # givens={
            #     self.X: data[self.index * self.batch_size: (self.index + 1) * self.batch_size],
            #     self.s: self.s[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            # }
        )
        # training time!
        for e in xrange(self.epochs):
            # idx = np.random.randint(0, n_batches)
            # c = train_function(idx)
            c = train_function(data)
        if verbose:
            print 'step', e, 'cost', c


class SparseCoder(object):

    def __init__(self, data, num_components,
                 learning_rate=0.01,
                 regularization_param=0.01,
                 sparsity_param=0.1,
                 sparsity_rate=0.01):
        # data = dictionary * code
        # X = W * s
        # both W and s will be learned, hence making them share
        self.data = data
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.sparsity_param = sparsity_param
        self.sparsity_rate = sparsity_rate
        self.num_components = num_components
        self.num_examples, self.num_features = data.shape
        self.patch_size = int(np.sqrt(self.num_features))
        self.X = T.matrix('x')
        self.W = theano.shared(value=np.random.uniform(-1, 1, (self.num_features, self.num_components)), name='W')
        self.s = theano.shared(value=np.zeros((self.num_components, self.num_examples)), name='s')
        # self.s = theano.shared(value=np.random.uniform(-1, 1, (self.num_components, self.num_examples)), name='s')

    @property
    def cost(self):
        squared_delta = T.mean((self.X - T.dot(self.W, self.s).T) ** 2)
        regularization = self.regularization_param * T.sum(self.W ** 2)
        sparsity = self.sparsity_rate * T.sum(T.sqrt(self.s ** 2 + self.sparsity_param))
        return squared_delta + sparsity + regularization

    @property
    def grad_dictionary(self):
        return T.grad(cost=self.cost, wrt=self.W)

    @property
    def grad_code(self):
        return T.grad(cost=self.cost, wrt=self.s)

    def run(self, epochs, visualize=False, visualize_dimensions=None):
        # first let's make two trainers
        cost = self.cost

        update_rule_dictionary = (
            (self.W, self.W - self.learning_rate * self.grad_dictionary),
        )
        dictionary_trainer = Trainer(
            self.X, update_rule_dictionary,
            cost, batch_size=100,
            learning_rate=self.learning_rate,
            epochs=500
        )

        update_rule_code = (
            (self.s, self.s - self.learning_rate * self.grad_code),
        )
        code_trainer = Trainer(
            self.X, update_rule_code,
            cost, batch_size=100,
            learning_rate=self.learning_rate,
            epochs=500
        )

        # c = theano.function(inputs=[self.X], outputs=self.cost)
        # print c(self.data)
        # and then train one by one
        if visualize:
            plt.ion()
        for e in xrange(epochs):
            print 'big iteration', e
            code_trainer.train(self.data, verbose=True)
            dictionary_trainer.train(self.data, verbose=True)
            if visualize:
                xdim, ydim = visualize_dimensions
                for i in xrange(self.num_components):
                    item = self.W.get_value()[:, i].reshape(self.patch_size, self.patch_size)
                    plt.subplot(xdim, ydim, i)
                    plt.xticks(())
                    plt.yticks(())
                    plt.imshow(item, cmap='gray', interpolation='none')
                plt.draw()
        if visualize:
            plt.ioff()


def whiten(X, fudge=1E-18):
    Xcov = np.dot(X.T, X)
    d, V = np.linalg.eigh(Xcov)
    D = np.diag(1. / np.sqrt(d + fudge))
    W = np.dot(np.dot(V, D), V.T)
    X_white = np.dot(X, W)
    return X_white


if __name__ == '__main__':
    patches = get_patches(8, 1000)
    patches = patches.reshape(patches.shape[0], -1)
    np.random.shuffle(patches)

    patches = whiten(patches)
    patches = preprocessing.scale(patches)
    # for i in xrange(100):
    #     plt.subplot(10, 10, i)
    #     plt.xticks(())
    #     plt.yticks(())
    #     plt.imshow(patches[i].reshape(8, 8), cmap='gray', interpolation='none')
    # plt.show()

    coder = SparseCoder(
        patches, 100,
        learning_rate=0.1,
        regularization_param=0.01,
        sparsity_rate=0.00005,
        sparsity_param=0.00001
    )
    coder.run(100, visualize=True, visualize_dimensions=(10, 10))
    raw_input()
    # test = theano.function(inputs=[coder.X], outputs=T.dot(coder.))
    # for i in xrange(10):
    #     idx = np.random.randint(0, 1000)
    #     p = patches[idx]
    #     code = np.