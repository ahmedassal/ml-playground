import numpy as np
import theano.tensor as T
import theano
from theano import config
from utils import get_mnist, read_lasagne_model, write_lasagne_model
from sklearn.cross_validation import train_test_split
import lasagne
from lasagne.nonlinearities import leaky_rectify, softmax, identity
from sklearn import preprocessing
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import pickle

theano.config.openmp = True
# theano.config.on_unused_input = 'ignore'
image_dim = 100
input_var = T.tensor4('X')
target_var = T.ivector('y')


def get_model(with_softmax=True):
    network = lasagne.layers.InputLayer((None, 1, image_dim, image_dim), input_var)
    network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')
    network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')
    network = lasagne.layers.Conv2DLayer(network, 128, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                        128, nonlinearity=leaky_rectify,
                                        W=lasagne.init.Orthogonal())
    if with_softmax:
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                            2, nonlinearity=softmax)
    else:
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                            2, nonlinearity=identity)
    return network


def train(data_train, target_train):
    network = get_model()

    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2)

    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                                momentum=0.9)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    write_lasagne_model(network, 'cnn_catdog')

    # train network (assuming you've got some training data in numpy arrays)
    batch_size = 10
    for epoch in range(200):
        loss = 0
        i = np.random.choice(range(data_train.shape[0] - batch_size))
        # for i in xrange(batches):
        input_batch = data_train[i: i + batch_size]
        target_batch = target_train[i: i + batch_size]
        input_batch = input_batch.reshape(batch_size, 1, image_dim, image_dim)
        loss += train_fn(input_batch, target_batch)
        print("Epoch %d: Loss %g" % (epoch + 1, loss / data_train.shape[0]))

    write_lasagne_model(network, 'cnn_catdog')


def test(network, data_test, target_test):
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    print 'success', np.count_nonzero(predict_fn(data_test.reshape(data_test.shape[0], 1, image_dim, image_dim)) == target_test) / float(data_test.shape[0])


def visualize(data_test, target_test):
    # get model without softmax
    network = get_model(with_softmax=False)
    # load params
    with open('cnn_catdog.model', 'r') as f:
        data = pickle.loads(f.read())
    # data[-1] = np.float32([1., 1.])
    lasagne.layers.set_all_param_values(network, data)

    out_unnormalized = lasagne.layers.get_output(network, deterministic=True)
    class_output = out_unnormalized[0][1]
    net_input = theano.shared(np.zeros((1, 1, image_dim, image_dim), dtype=theano.config.floatX))
    # cost = class_output - T.mean(input_var ** 2) * 0.001
    cost = class_output
    func = theano.function(
        inputs=[],
        outputs=[class_output],
        updates=((net_input, net_input + 1 * theano.grad(cost, wrt=input_var)),),
        givens={
            input_var: net_input
        }
    )
    for i in xrange(200):
        print i, func()
    img = net_input.get_value()[0, 0, :, :]
    plt.imshow(img, cmap='gray') #, interpolation='none')
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    # data, target = get_mnist(random=True, num=2000)
    # data = np.float32(data)
    # data = preprocessing.scale(data)
    # target = np.int32(target)

    data = np.float32(np.load('catdog_data.npy'))
    data = np.float32(np.vstack([resize(d.reshape(224, 224), (image_dim, image_dim)).ravel() for d in data]))
    data = preprocessing.scale(data)
    print 'resizedok'
    target = np.int32(np.load('catdog_labels.npy'))

    (
        data_train, data_test,
        target_train, target_test
    ) = train_test_split(data, target, test_size=0.2)


    # visualize(data_test, target_test)
    train(data_train, target_train)

    network = get_model()
    read_lasagne_model(network, 'cnn_catdog')
    test(network, data_test, target_test)
