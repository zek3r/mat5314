import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


# Polynomial kernel matrix ---------------------------------------------------------------------------------------------

def poly_kernel(X1, X2, l):
    """
    :param X1:  data matrix 1
    :param X2:  data matrix 2
    :param l:  power of polynomial kernel
    :return K: kernel matrix
    """

    m = X1.shape[0]  # rows
    n = X2.shape[0]  # cols

    K = np.zeros((m, n))  # Kernel matrix

    for i in range(m):
        for j in range(n):
            K[i, j] = (np.dot(X1[i, :], X2[j, :])+1)**l

    return K


# Linear kernel matrix -------------------------------------------------------------------------------------------------

def lin_kernel(X1, X2):
    """
    :param X1:  data matrix 1
    :param X2:  data matrix 2
    :return K: kernel matrix
    """

    m = X1.shape[0]  # rows
    n = X2.shape[0]  # cols

    K = np.zeros((m, n))  # Kernel matrix

    for i in range(m):
        for j in range(m):
            K[i, j] = np.dot(X1[i, :], X2[j, :])

    return K


# Regularized Least Squares train --------------------------------------------------------------------------------------

def rls_train(Ytrain, Kmat, gamma):
    """
    :param Ytrain: labels
    :param Kmat: data matrix
    :param gamma: lasso coefficient
    :return w:    vector of coefficients
    :return model: fitted model
    """
    m = len(Ytrain)
    A = Kmat + m*gamma*np.eye(m)
    w = np.linalg.solve(A, Ytrain)

    return w


# Regularized Least Squares predict ------------------------------------------------------------------------------------

def rls_predict(w, Knew):
    """
    :param w:  fitted model
    :param Knew:  Kmat for new data
    :return Y_pred: predictions for new data
    """
    Y_pred = np.matmul(Knew.transpose(), w)

    return Y_pred


# Cross Validate -------------------------------------------------------------------------------------------------------

def cross_validate(Xtrain, Xtest, Ytrain, Ytest, kernel):
    """
    :param Xtrain:  data matrix
    :param Ytrain: labels
    :param Xtest:  test data
    :param Ytest:  test labels
    :param kernel: tuple, kernel type. Options are 'linear' or ['polynomial', l]
    :return w:    vector of coefficients
    :return Y_pred: predicted labels
    :return sse:    error
    :return gammas: gammas used
    """

    if type(kernel) is str:
        kernel = (kernel,)

    if kernel[0] == 'linear':
        Kmat = lin_kernel(Xtrain, Xtrain)
        Kmat_test = lin_kernel(Xtrain, Xtest)
    elif kernel[0] == 'polynomial' and (type(kernel[1]) == int or type(kernel[1] == float)):
        Kmat = poly_kernel(Xtrain, Xtrain, kernel[1])
        Kmat_test = poly_kernel(Xtrain, Xtest, kernel[1])
    else:
        print('Invalid Kernel Type')
        sys.exit(1)

    #max_eig = int(np.max(np.real(np.linalg.eig(Kmat)[0])))
    max = 100

    gammas = np.arange(1, max, 0.5)
    itr = len(gammas)

    w = list(np.zeros(itr))
    Y_pred = list(np.zeros(itr))
    test_error = list(np.zeros(itr))
    test_sse = np.zeros(itr)
    sse = np.zeros(itr)

    for i, g in enumerate(gammas):
        w[i] = rls_train(Ytrain, Kmat, g)

        Y_pred[i] = np.sign(rls_predict(w[i], Kmat_test))
        sse[i] = np.sum((Y_pred[i] - Ytest) ** 2)

        test_error[i] = np.sign(rls_predict(w[i], Kmat))
        test_sse[i] = np.sum((test_error[i] - Ytrain) ** 2)

    return w, Y_pred, sse, gammas, test_sse


# Main Script ----------------------------------------------------------------------------------------------------------

data = sio.loadmat('2Moons-2.mat')
Xtrain = data.get('x')
Xtest = data.get('xt')
Ytrain = data.get('y')
Ytest = data.get('yt')

# Evaluate Linear model
kernel = 'linear'
w_l, Y_pred_l, sse_l, gammas_l, test_sse_l = cross_validate(Xtrain, Xtest, Ytrain, Ytest, kernel)

best_model_index_l = np.argmin(sse_l)
best_gamma_l = gammas_l[best_model_index_l]
model_error_l = np.min(sse_l)

# Evaluate polynomial model
params = [3, 5, 8]
w_p = list(np.zeros(3))
Y_pred_p = list(np.zeros(3))
sse_p = list(np.zeros(3))
gammas_p = list(np.zeros(3))
best_model_index_p = list(np.zeros(3))
best_gamma_p = list(np.zeros(3))
model_error_p = list(np.zeros(3))
test_sse_p = list(np.zeros(3))

for i, l in enumerate(params):
    w_p[i], Y_pred_p[i], sse_p[i], gammas_p[i], test_sse_p[i] = \
        cross_validate(Xtrain, Xtest, Ytrain, Ytest, kernel=('polynomial', params[i]))

    best_model_index_p[i] = np.argmin(sse_p[i])
    best_gamma_p[i] = gammas_p[i][best_model_index_p[i]]
    model_error_p[i] = np.min(sse_p[i])

# Display results and plot
print('Error for linear model with gamma '+str(best_gamma_l)+' is: '+str(model_error_l))
for i in range(len(params)):
    print('Error for order '+str(params[i])+' polynomial model with gamma '+str(best_gamma_p[i])+' is: '+str(model_error_p[i]))

# Plot data
plt.figure(1)
Xtotal = np.concatenate((Xtest, Xtrain), axis=0)
Ytotal = np.concatenate((Ytest, Ytrain), axis=0)
plt.title('All Data')
plt.scatter(Xtotal[(Ytotal == -1).reshape(400,), 0], Xtotal[(Ytotal == -1).reshape(400,), 1], color='blue')
plt.scatter(Xtotal[(Ytotal == 1).reshape(400,), 0], Xtotal[(Ytotal == 1).reshape(400,), 1], color='red')

#plt.savefig('/Users/macbookair/desktop/MAT5314/tex_stuff/fig1.pdf',format='pdf')

fig = plt.figure(2, tight_layout=True)
gs0 = gs.GridSpec(2, 2)

# Plot test data for linear
index = (Y_pred_l[best_model_index_l] == 1).reshape(200,)
Xplus = Xtest[index, :]
index = (Y_pred_l[best_model_index_l] == -1).reshape(200,)
Xminus = Xtest[index, :]

ax1 = fig.add_subplot(gs0[0,0])
ax1.set_title('Linear Kernel')
ax1.scatter(Xplus[:, 0], Xplus[:, 1], color='red')
ax1.scatter(Xminus[:, 0], Xminus[:, 1], color='blue')

# Plot test data for poly
ind = [[0, 1], [1, 0], [1, 1]]
titles = ['Polynomial, order='+str(params[0]), 'Polynomial, order='+str(params[1]), 'Polynomial, order='+str(params[2])]
for i in range(len(best_model_index_p)):
    ax2 = fig.add_subplot(gs0[ind[i][0],ind[i][1]])
    ax2.set_title(titles[i])
    index = (Y_pred_p[i][best_model_index_p[i]] == 1).reshape(200,)
    Xplus = Xtest[index, :]
    index = (Y_pred_p[i][best_model_index_p[i]] == -1).reshape(200,)
    Xminus = Xtest[index, :]

    ax2.scatter(Xplus[:, 0], Xplus[:, 1], color='red')
    ax2.scatter(Xminus[:, 0], Xminus[:, 1], color='blue')

#plt.savefig('/Users/macbookair/desktop/MAT5314/tex_stuff/fig2.pdf',format='pdf')

# Plot error

plt.figure(3)
colours = ['red', 'orange', 'green']
plt.plot(gammas_l, sse_l, color='black')
plt.title('Cross Validation Error as Function of Reg Param')
for i in range(len(best_model_index_p)):
    plt.plot(gammas_p[i], sse_p[i], color=colours[i])

plt.savefig('/Users/macbookair/desktop/MAT5314/tex_stuff/fig3.pdf',format='pdf')

plt.figure(4)
plt.plot(gammas_l, test_sse_l, color='black')
plt.title('Training Set Error as Function of Reg Param')
for i in range(len(best_model_index_p)):
    plt.plot(gammas_p[i], test_sse_p[i], color=colours[i])

plt.savefig('/Users/macbookair/desktop/MAT5314/tex_stuff/fig4.pdf',format='pdf')



