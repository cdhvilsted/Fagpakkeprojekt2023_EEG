###################### group ICA algoritm for EEG data ########################
###############################################################################

# packages
import numpy as np
import mne
import matplotlib.pyplot as plt
import picard as pic
#import fast ICA
from sklearn.decomposition import FastICA

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
def g(x):
    return np.tanh(x)
def g_der(x):
    return 1 - g(x) * g(x)
def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new

def ICA(X, R, G, typeICA):
    # this function takes a matrix and returns the ICA of the matrix
    if typeICA == "fastICA":
        ica = FastICA(n_components=10)
        S = ica.fit_transform(X)  # Reconstruct signals
        A = ica.mixing_  # Get estimated mixing matrix

    # picard is the fastest!!!!!!!!!!!! :)
    elif typeICA == "picard":
        K, W, Y = pic.Picard(X, whiten = False)

        w = np.dot(W, G)
        A = np.dot(w.T, np.linalg.inv(np.dot(w, w.T)))

    else:
        # Throw error
        print("Error: type must be either 'fastICA' or 'picard'")
        return



    return S, A, W















