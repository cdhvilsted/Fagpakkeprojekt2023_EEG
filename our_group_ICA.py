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

    reduced_X = U[:,:reduced_dim] @ np.diag(S[:reduced_dim])


    # if you want to plot the variance explained
    if plot:
        plotCumulativeExplainedVariances(rho)

    return U, S, V, reduced_X, rho


def _g(x):
    return np.tanh(x)

def _gprime(x):
    return 1 - np.tanh(x)**2

def ICA(X, R, G, typeICA):
    # this function takes a matrix and returns the ICA of the matrix
    if typeICA == "fastICA":
        max_iter = 200
        tol = 1e-4
        n_samples, n_features = X.shape
        W = np.zeros((140, n_features), dtype=X.dtype)
        # Main loop
        for i in tqdm(range(140)):
            w = np.random.randn(10)
            for j in range(max_iter):
                wx = np.dot(w, X.T)
                gwtx = _g(wx).mean()
                g_wtx = _gprime(wx)
                w_new = (X * g_wtx).T @ wx / n_samples - g_wtx.mean() * w
                w_new = w_new / np.sqrt((w_new ** 2).sum())
                if i > 0:
                    w_new = w_new - np.dot(np.dot(w_new, W[:i].T), W[:i])
                lim = np.abs(np.abs((w * w_new).sum()) - 1)
                w = w_new
                if lim < tol:
                    break
        W[i, :] = w
        S = np.dot(W, X)

    else:
        # Throw error
        print("Error: type must be 'fastICA'")
        return

    # reconstruct the signals
    A = np.linalg.pinv(W)

    return S, A, W















