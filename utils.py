from sklearn.datasets import fetch_mldata
import numpy as np
import mdp
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_sample_images
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import ndimage as ndi
from skimage.transform import resize
import cPickle as pickle
import os
import lasagne


def get_gabors(side):
    sgm = (2., 1.)
    freq = 1. / 10
    phi = np.pi / 2.
    gabors = np.empty((180 / 10, side * side))
    for i, angle in enumerate(range(0, 180, 10)):
        alpha = np.radians(angle)
        gb = mdp.utils.gabor((side, side), alpha, phi, freq, sgm).ravel()
        gabors[i, :] = MinMaxScaler(feature_range=(-1., 1.)).fit_transform(gb)
    return gabors


def get_gabor(side):
    g = mdp.utils.gabor(
        (20, 20), 0, np.pi / 2.,
        0.5, (4., .5)
    )
    return resize(g, (side, side))


def get_mnist(start=None, end=None, random=False, num=None):
    mnist = fetch_mldata('MNIST original', data_home='~/diss/mnist')
    if random is not None and num is not None:
        idx = np.random.choice(range(mnist.data.shape[0]), num)
    elif start is not None and end is not None:
        idx = range(start, end)
    return mnist.data[idx], mnist.target[idx]


def get_patches(size, num=5000):
    imgs = load_sample_images()
    img = (np.float64(imgs.images[1]) / 255.).mean(axis=2)
    # img = ndi.gaussian_filter(img, .5) - ndi.gaussian_filter(img, 1)
    return extract_patches_2d(img, (size, size), max_patches=num)


def binarize(_arr, min_value=-1, max_value=1):
    arr = np.float32(_arr)
    arr[arr <= 0] = min_value
    arr[arr > 0] = max_value
    return arr


def sigmoid(x):
    return 1.0 / (1. + np.exp(-x))


def read_lasagne_model(model, filename):
    filename = os.path.join('./', '%s.%s' % (filename, 'model'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_lasagne_model(model, filename):
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'model')
    with open(filename, 'w') as f:
        pickle.dump(data, f)
