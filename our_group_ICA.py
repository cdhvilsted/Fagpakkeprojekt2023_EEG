###################### group ICA algoritm for EEG data ########################
###############################################################################

# packages
import numpy as np
import mne
import matplotlib.pyplot as plt
from picard import picard
#import fast ICA
from sklearn.decomposition import FastICA
from tqdm import tqdm
import time

###############################################################################




def plotCumulativeExplainedVariances(rho, threshold = 0.95):
    threshold = 0.9
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()

def center_data(X):
    # this function centers the data
    # input: X: matrix
    # output: centered matrix
    return X - np.ones((len(X), 1)) * X.mean(axis=0)

def SVD(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T  # transpose Vt to obtain V

    return U, S, V

def PCA(X,reduced_dim, plot = True):
    # this function takes a matrix and returns the PCA of the matrix
    # input: X: matrix
    # output: PCA of the matrix
    # reduced_dim: number of dimensions to reduce to

    # center the data
    X_tilde = center_data(X)

    # PCA by computing SVD of Y
    U, S, V = SVD(X_tilde)


    #explained variances
    rho = (S * S) / (S * S).sum()

    #reduced_X = U[:,:reduced_dim] @ np.diag(S[:reduced_dim])
    reduced_X = X_tilde @ V[:,:reduced_dim] 

    # if you want to plot the variance explained
    if plot:
        plotCumulativeExplainedVariances(rho)

    return U, S, V, reduced_X, rho

import sklearn
def _g(x):
    return np.tanh(x)

def _gprime(x):
    return 1 - np.tanh(x)**2

def ICA(X, R, G, typeICA):
    # this function takes a matrix and returns the ICA of the matrix
    if typeICA == "fastICA":
        # fastICA
        transform_ICA = sklearn.decomposition.FastICA(n_components = None,  whiten=False, fun='logcosh', max_iter=200, tol=0.0001)

        # fit the model
        S = transform_ICA.fit_transform(X)
        A = transform_ICA.mixing_

        W = np.linalg.pinv(A)

        #pvaf
        print(pvaf(X,S))

        explained = np.var(S, axis = 0)
        explained_ratio = explained / np.sum(explained)
        sorted = np.argsort(explained_ratio)
        print(np.max(explained))
        print(sum(explained))
        print(sorted)

    elif typeICA == "picard":
        # use picard
        W, S = picard(
            X_concat,
            ortho=ortho,
            extended=extended,
            centering=False,
            max_iter=max_iter,
            whiten= False,
            tol=tol,
            random_state=random_state,
        )
        A = np.linalg.pinv(W)

    else:
        # Throw error
        print("Error: type must be 'fastICA'")
        return

    # reconstruct the signals

    return S, A, W, sorted

def pvaf(X,S):
    pvaf = []
    for i in range(140):
        pvaf.append(100-100*np.mean(np.var(np.transpose(X)-S[:,i]))/np.mean(np.var(X)))
    print("pvaf: ", pvaf, 'pvaf_sum', np.sum(pvaf))
