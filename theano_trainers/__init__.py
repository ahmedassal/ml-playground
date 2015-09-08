import theano.tensor as T


class BaseTrainer(object):

    def __init__(self, input, output, params, cost, learning_rate=0.1, epochs=1000):
        self.input = input
        self.output = output
        self.params = params  # (Weight1, bias1, Weight2, bias2...)
        self.cost = cost
        self.learning_rate = learning_rate
        self.epochs = epochs

    def get_update_rules(self):
        rules = []
        for w, b in zip(self.params[:-1:2], self.params[1::2]):  # iterating over (weight, bias) pairs
            rules.append((w, w - self.learning_rate * T.grad(cost=self.cost, wrt=w)))
            rules.append((b, b - self.learning_rate * T.grad(cost=self.cost, wrt=b)))
        return rules
