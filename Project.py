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
    Initializes connectivity matrix W
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
    Initializes bias vector b
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


def softmax(z, axis=0):
    """
    Generates softmax of input vector z
    :param z: input array
    :param axis: which axis to calculate softmax over (0=col, 1=row) if z is 2d
    :return: softmax of z
    """

    if len(z.shape) == 1:
        s = np.exp(z)/np.sum(np.exp(z))

    if len(z.shape) == 2:
        s = np.zeros(z.shape)
        if axis == 0:
            for i in range(z.shape[1]):
                s[:, i] = np.exp(z[:, i])/np.sum(np.exp(z[:, i]))
        if axis == 1:
            for i in range(z.shape[0]):
                s[i, :] = np.exp(z[i, :])/np.sum(np.exp(z[i, :]))


    return s


def lnn_likelihood(W, b, X, y, out_vals):
    """
    Calculates log likelihood of linear neural network for input data X and target values y
    :param W: connectivity matrix d x out
    :param b: bias vector (must be 1d)
    :param X: input data matrix. This should be d x m
    :param y: input target values. This should be m x 1
    :param out_vals: list of distinct output values. This should be out x 1
    :return: dW: derivatives of weight matrix
    :return: db: derivatives of bias vector
    """

    m = len(y)  # num_samples
    o = len(out_vals)  # num_outputs

    b = np.tile(b, (m, 1)).transpose()
    z = W.transpose() @ X + b

    s_z = softmax(z, axis=0)

    db = np.zeros(o)
    dW = np.zeros(W.shape)
    for j in range(o):
        m_j = np.sum(np.isin(y, out_vals))

        db[j] =  m_j + np.sum(s_z[j, :])
        dW[:, j] =

def lnn_predict(W, b, X, y):
    """
    predicts values of linear neural network for test data X
    :param W: connectivity matrix
    :param b: bias vector (must be 1d)
    :param X: input data matrix. This should be d x m
    :return: y_hat: predicted values
    """

    num_samples = X.shape[1]
    b = np.tile(b, (num_samples, 1)).tranpose()

    z = W.transpose() @ X + b

    y_hat = softmax(z, axis=0)

    return y_hat





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


