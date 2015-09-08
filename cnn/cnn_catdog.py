import numpy as np
import theano.tensor as T
import theano
from theano import config
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano_trainers.mgd import MiniBatchGradientDescentTrainer
from utils import get_mnist
import os
import time
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from skimage.transform import resize
from pilkit.processors import SmartCrop, ResizeToFit
from PIL import Image


# theano.config.optimizer = 'fast_compile'
theano.config.openmp = True
theano.config.openmp_elemwise_minsize = 50000


def load_data(num=2000, resize=256, crop=(224, 224),
              folder='catdog_data/train'):
    files = os.listdir(folder)
    chosen = np.random.choice(files, num)
    data = []

    # processors
    w, h = crop
    cropper = SmartCrop(width=w, height=h)

    def get_new_dimensions(im):
        w, h = im.size
        if w < h:
            orientation = 'v'
            small, big = w, h
        else:
            orientation = 'h'
            small, big = h, w
        small_size = resize
        big_size = int(big * float(small_size) / small)
        if orientation == 'v':
            return small_size, big_size
        else:
            return big_size, small_size
    labels = []
    for i, fname in enumerate(chosen):
        print i
        path = os.path.join(folder, fname)
        im = Image.open(path)
        # first resize
        w, h = get_new_dimensions(im)
        resizer = ResizeToFit(width=w, height=h, upscale=True)
        im = resizer.process(im)
        # then crop
        im = cropper.process(im)
        # then to numpy and rescale to [0, 1]
        im = np.float32(np.mean(np.asarray(im), axis=-1)) / 255.
        data.append(im.ravel())
        # don't forget labels
        if 'cat' in fname:
            labels.append(0)
        else:
            labels.append(1)
    return np.vstack(data), np.hstack(labels)


def convnet_layer(input, w_shape, img_shape, pool_size):
    # dimensions
    input_size = np.prod(w_shape[1:])
    output_size = np.prod(w_shape[0] * np.prod(w_shape[2:])) / np.prod(pool_size)
    w_bound = np.sqrt(6. / (input_size + output_size))

    # weights
    w = theano.shared(np.random.uniform(low=-w_bound, high=w_bound, size=w_shape).astype(config.floatX))
    b = theano.shared(np.zeros((w_shape[0],), dtype=config.floatX))

    # convolve
    convolved = conv.conv2d(input=input, filters=w, filter_shape=w_shape, image_shape=img_shape)
    # pool
    pooled = downsample.max_pool_2d(input=convolved, ds=pool_size, ignore_border=True)

    # activation
    return T.nnet.sigmoid(pooled + b.dimshuffle('x', 0, 'x', 'x')), w, b

if __name__ == '__main__':
    image_dim = 64

    data = np.float32(np.load('catdog_data.npy'))
    data = np.float32(np.vstack([resize(d.reshape(224, 224), (image_dim, image_dim)).ravel() for d in data]))
    data = preprocessing.scale(data)
    target = np.int32(np.load('catdog_labels.npy'))
    (
        data_train, data_test,
        target_train, target_test
    ) = train_test_split(data, target, test_size=0.2)
    test_size = data_test.shape[0]

    # data_train = data_train.reshape(
    #     data_train.shape[0],
    #     1,
    #     image_dim, image_dim
    # )
    # data_test = data_test.reshape(
    #     data_test.shape[0],
    #     1,
    #     image_dim, image_dim
    # )

    # data, target = get_mnist('../datasets/mnist', 2000)

    # conv params
    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = 10

    input = x.reshape((batch_size, 1, image_dim, image_dim))

    # conv layers
    # lvl1
    lvl1_filter_dim = 5
    lvl1_pool_dim = 2
    a1, w1, b1 = convnet_layer(
        input, w_shape=(16, 1, lvl1_filter_dim, lvl1_filter_dim),
        img_shape=(batch_size, 1, image_dim, image_dim),
        pool_size=(lvl1_pool_dim, lvl1_pool_dim)
    )
    # lvl2
    lvl2_image_dim = (image_dim - lvl1_filter_dim + 1) / lvl1_pool_dim
    lvl2_filter_dim = 5
    lvl2_pool_dim = 2
    a2, w2, b2 = convnet_layer(
        a1, w_shape=(32, 16, lvl2_filter_dim, lvl2_filter_dim),
        img_shape=(batch_size, 16, lvl2_image_dim, lvl2_image_dim),
        pool_size=(lvl2_pool_dim, lvl2_pool_dim)
    )
    # lvl3
    lvl3_image_dim = (lvl2_image_dim - lvl2_filter_dim + 1) / lvl2_pool_dim
    lvl3_filter_dim = 5
    lvl3_pool_dim = 2
    a3, w3, b3 = convnet_layer(
        a2, w_shape=(64, 32, lvl3_filter_dim, lvl3_filter_dim),
        img_shape=(batch_size, 32, lvl3_image_dim, lvl3_image_dim),
        pool_size=(lvl3_pool_dim, lvl3_pool_dim)
    )

    a3 = a3.flatten(2)

    # mpl on top. hidden layer
    fc_image_dim = (lvl3_image_dim - lvl3_filter_dim + 1) / lvl3_pool_dim
    n_hidden = 100
    w4 = theano.shared(value=np.random.uniform(-0.7, 0.7, (fc_image_dim * fc_image_dim * 64, n_hidden)).astype(config.floatX))
    b4 = theano.shared(value=np.zeros((n_hidden,), dtype=config.floatX))
    a4 = 1 / (1 + T.exp(-T.dot(a3, w4) - b4))

    # and the softmax
    n_classes = 2
    w5 = theano.shared(value=np.random.uniform(-0.7, 0.7, (n_hidden, n_classes)).astype(config.floatX))
    b5 = theano.shared(value=np.zeros((n_classes,), dtype=config.floatX))
    softmax = T.nnet.softmax(T.dot(a4, w5) + b5)

    w_dec = 0.001
    cost = -T.mean(T.log(softmax)[T.arange(y.shape[0]), y]) + w_dec * (T.sum(w1 ** 2) + T.sum(w2 ** 2) + T.sum(w3 ** 2) + T.sum(w4 ** 2) + T.sum(w5 ** 2))

    params = (w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)

    MiniBatchGradientDescentTrainer(
        x, y, params, cost, learning_rate=0.1,
        batch_size=batch_size, epochs=2000
    ).train(data_train, target_train)
    # GradientDescentTrainer(x, y, params, cost, epochs=5000).train(data, target)

    # checking result

    data_test = theano.shared(value=data_test)

    hypothesis = theano.function([x], T.argmax(softmax, axis=1))
    index = T.iscalar('i')
    hypothesis = theano.function(
        inputs=[index],
        outputs=T.argmax(softmax, axis=1),
        givens={
            x: data_test[index * batch_size: (index + 1) * batch_size],
        }
    )
    print 'here'

    # test
    batches = test_size / batch_size
    if batches * batch_size < test_size:
        batches += 1
    matches = 0
    for i in xrange(batches):
        # print i
        sub_target = target_test[i * batch_size: (i + 1) * batch_size]
        sub_hypothesis = hypothesis(i)
        matches += np.count_nonzero(sub_target == sub_hypothesis)
    print
    print 'success', float(matches) / test_size
