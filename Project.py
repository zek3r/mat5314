from pathlib import Path
import numpy as np
import scipy as sp
import scipy.io as sio
import keras as k
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# Functions-------------------------------------------------------------------------------------------------------------


def load_mnist(name):
    """
    Function loads MNIST data set. Assumes that the data file is contained in the working directory
    :param name: string, name of file including extension
    :return: data: MNIST data set.
    """
    working_dir = Path.cwd()
    get_file = working_dir / name
    data = sio.loadmat(get_file)

    return data


def init_mat(element_sd, p_connection, num_features, num_outputs, spec_radius=None):
    """
    :param element_sd: Standard deviation of each element
    :param p_connection: probability that a single element is non-zero
    :param num_features: (d) number of features (size of input layer)
    :param num_outputs: (o) number of outputs (size of output layer
    :param spec_radius: desired spectral radius for matrix
    :return: W: d x o connectivity matrix
    """

    W = np.random.binomial(1, p_connection, (num_features, num_outputs))
    X = np.random.normal(0, element_sd, (num_features, num_outputs))
    W = W * X

    if spec_radius:
        rho = np.max(np.linalg.eigvals(W))
        W = spec_radius*W/rho

    return W


def init_bias(num_outputs, element_mean=None, element_sd=None):
    """
    :param num_outputs: (o) size of bias vector
    :param element_mean: mean of elements. If None mean is zero
    :param element_sd: sd of elements. If None vector is constant. If both params are None then zero vector returned
    :return: bias vector
    """
    if element_mean and element_sd:
        b = np.random.normal(element_mean, element_sd, num_outputs)
    elif element_mean:
        b = np.ones(num_outputs)*element_mean
    elif element_sd:
        b = np.random.normal(0, element_sd, num_outputs)
    else:
        b = np.zeros(num_outputs)

    return b


# Main Script-----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    data = load_mnist('mnist_all.mat')                      # Load

    to_plot = []
    for key, value in data.items():                         # Rescale
        if key[0] == 't':
            value = value/255.0
            if key[1] == 'r':
                to_plot.append(value[0].reshape(28, 28))

    fig = plt.figure(1)                                     # Plot numbers
    gs0 = gs.GridSpec(2, 5)
    index = [(x, y) for x in range(2) for y in range(5)]

    for i, ix in enumerate(index):
        ax = fig.add_subplot(gs0[ix])
        ax.imshow(to_plot[i], cmap='Greys')
        ax.axis('off')


