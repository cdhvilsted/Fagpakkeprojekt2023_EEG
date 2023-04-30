###################### group ICA algoritm for EEG data ########################
###############################################################################

# packages
import numpy as np
import mne
import matplotlib.pyplot as plt

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



def ICA(X, U, S, V):
# this function takes a matrix and returns the ICA of the matrix
    # input: X: matrix
    # output: ICA of the matrix

    # center the data
    Y = center_data(X)


    # compute the whitening matrix
    W = np.diag(1 / S) @ U.T

    # compute the unmixing matrix
    A = V.T @ np.diag(S)

    # compute the ICA
    Z = W @ Y

    return Z, W, A















