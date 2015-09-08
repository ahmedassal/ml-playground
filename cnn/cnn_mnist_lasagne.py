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
input_var = T.tensor4('X')
target_var = T.ivector('y')
image_dim = 28


def get_model(with_softmax=True):
    # create a small convolutional neural network
    network = lasagne.layers.InputLayer((None, 1, image_dim, image_dim), input_var)
    network = lasagne.layers.Conv2DLayer(network, 64, (5, 5),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')
    network = lasagne.layers.Conv2DLayer(network, 32, (5, 5),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                        128, nonlinearity=leaky_rectify,
                                        W=lasagne.init.Orthogonal())
    if with_softmax:
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                            10, nonlinearity=softmax)
    else:
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                            10, nonlinearity=identity)
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

    # train network (assuming you've got some training data in numpy arrays)
    batch_size = 20
    for epoch in range(1000):
        loss = 0
        i = np.random.choice(range(data_train.shape[0] - batch_size))
        input_batch = data_train[i: i + batch_size]
        target_batch = target_train[i: i + batch_size]
        input_batch = input_batch.reshape(batch_size, 1, image_dim, image_dim)
        loss += train_fn(input_batch, target_batch)
        print("Epoch %d: Loss %g" % (epoch + 1, loss / data_train.shape[0]))

    write_lasagne_model(network, 'cnn_mnist')
    return network


def test(network, data_test, target_test):
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    print 'success', np.count_nonzero(predict_fn(data_test.reshape(data_test.shape[0], 1, image_dim, image_dim)) == target_test) / float(data_test.shape[0])


def visualize():
    # get model without softmax
    network = get_model(with_softmax=False)
    # load params
    with open('cnn_mnist.model', 'r') as f:
        data = pickle.loads(f.read())
    # data[-1] = np.float32([1.] * 10)
    print data[-1]
    lasagne.layers.set_all_param_values(network, data)

    out_unnormalized = lasagne.layers.get_output(network, deterministic=True)

    imgs = []
    for c in range(10):
        class_output = out_unnormalized[0][c]
        net_input = theano.shared(np.zeros((1, 1, image_dim, image_dim), dtype=theano.config.floatX))
        # cost = (class_output - T.mean(input_var ** 2) * 0.001)
        cost = class_output
        pred = theano.function(
            inputs=[],
            outputs=out_unnormalized,
            givens={
                input_var: net_input
            }
        )
        func = theano.function(
            inputs=[],
            outputs=[cost],
            updates=((net_input, net_input + 1 * theano.grad(cost, wrt=input_var)),),
            givens={
                input_var: net_input
            }
        )
        for i in xrange(1000):
            print i, func()
        # print pred()
        img = net_input.get_value()[0, 0, :, :]
        imgs.append(img)

    for i, img in enumerate(imgs):
        plt.subplot(2, 5, i)
        plt.imshow(img, cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()



if __name__ == '__main__':
    data, target = get_mnist(random=True, num=2000)
    data = np.float32(data)
    target = np.int32(target)
    data = preprocessing.scale(data)
    (
        data_train, data_test,
        target_train, target_test
    ) = train_test_split(data, target, test_size=0.2)

    visualize()

    # network = train(data_train, target_train)
    # network = get_model()
    # read_lasagne_model(network, 'cnn_mnist')
    # test(network, data_test, target_test)
