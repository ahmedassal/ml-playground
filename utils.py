from sklearn.datasets import fetch_mldata


def get_mnist(start, end):
    mnist = fetch_mldata('MNIST original', data_home='~/diss/mnist')
    randidx = range(start, end)
    return mnist.data[randidx], mnist.target[randidx]
