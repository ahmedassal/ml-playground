import theano
import theano.tensor as T
import numpy as np

from theano_trainers import BaseTrainer


class MiniBatchGradientDescentTrainer(BaseTrainer):

    """
    With mini-batches.
    """

    def __init__(self, input, output, params, cost, batch_size, learning_rate=0.001, epochs=1000):
        super(MiniBatchGradientDescentTrainer, self).__init__(input, output, params, cost, learning_rate, epochs)
        self.batch_size = batch_size
        self.index = T.iscalar('i')

    def train(self, data, target):
        # train function now is a bit tricky and takes index as input
        # but updates are the same
        updates = self.get_update_rules()
        n_batches = data.shape[0] / self.batch_size
        # now to be able to make subtensors we need to make our data shared values
        data = theano.shared(value=data)
        target = theano.shared(value=target)
        train_function = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=updates,
            givens={
                self.input: data[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.output: target[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        # training time!
        for e in xrange(self.epochs):
            idx = np.random.randint(0, n_batches)
            c = train_function(idx)
            print 'step', e, 'cost', c
            # for i in xrange(n_batches):
            #     c = train_function(i)
            #     print c
        print 'training done'
