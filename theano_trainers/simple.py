import theano

from theano_trainers import BaseTrainer


class GradientDescentTrainer(BaseTrainer):

    """
    Plain and simple GD, no minibatches / stochasticity / cross-validation.

    NOTE: trying to make a general trainer class suitable for different kinds
    of models. Therefore `params` is an arbitrary length set of weights/biases
    (just a tuple, basically).
    """

    def train(self, data, target):
        updates = self.get_update_rules()
        train_function = theano.function(
            inputs=[self.input, self.output],
            outputs=self.cost,
            updates=updates
        )
        # training time!
        for i in xrange(self.epochs):
            print i
            train_function(data, target)
        print 'training done'
