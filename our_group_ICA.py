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
from ICA_dataImport import common, montage
import pandas as pd

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
    S = np.diag(S)
    
    if (U @ S @ Vt != X).all():
        print('Warning: U S Vt != X')
    
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

def componentPlot(R, numberComponents, numberSubjects):
    biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    to_drop_ch = list(set(montage.ch_names) - set(common))
    fig, ax = plt.subplots(numberComponents, numberSubjects, figsize=(20, 7))
    axs = ax.ravel()
    pbar = tqdm(total=numberComponents * numberSubjects)  # Initialize the progress bar
    count = 0
    minR = round(np.min(R),3)
    maxR = round(np.max(R),3)

    for i in range(numberComponents):
        for j in range(numberSubjects):
            data = R[i, :, j].tolist()
            df = pd.DataFrame([data], columns=common)
            df[to_drop_ch] = 0
            #df = df*1e-6
            df = df.reindex(columns=montage.ch_names)
            info = mne.create_info(ch_names=montage.ch_names, sfreq=10, ch_types='eeg')
            comp1 = mne.EvokedArray(df.to_numpy().T, info)
            comp1.set_montage(montage)
            comp1 = comp1.drop_channels(to_drop_ch)
            im = comp1.plot_topomap(times=[0], axes=axs[count], colorbar=False, show=False, sphere=0.12)
            pbar.update(1)  # Update the progress bar
            ax[i, j].set_title(' ')
            ax[0, j].set_xlabel('Subject ' + str(j))
            ax[i, 0].set_ylabel('Component ' + str(i))
            ax[0, j].xaxis.set_label_position('top')
            count += 1
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r'), ax=axs.ravel().tolist(),fraction=0.047*7/20)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([minR, 0, maxR])
    plt.show()


def timeSeriesPlot(U_3d, component, subject):
    U3 = U_3d[component, :, subject]
    U3 = U3.reshape(97, 141)
    U3 = np.mean(U3, axis=0)
    plt.gca().invert_yaxis()
    plt.plot(np.arange(-0.1, 1, step=1 / 128), U3)
    plt.show()

def timeSeriesPlotICA(U_3d, component):
    U3 = U_3d[component, :]
    U3 = U3.reshape(97, 141)
    U3 = np.mean(U3, axis=0)
    plt.gca().invert_yaxis()
    plt.plot(np.arange(-0.1, 1, step=1 / 128), U3)
    plt.show()


'''
def ComponentPlot(R_3d, numberComponents, numberSubjects):
    biosemi_montage = mne.channels.make_standard_montage('standard_1020',head_size=0.15)
    to_drop_ch = list(set(montage.ch_names)-set(common))

    fig, ax = plt.subplots(14,10, figsize=(15,12))
    axs = ax.ravel()
    count = 0
    for j in range(numberSubjects):
        for i in range(numberComponents):
            # make back_Y a list
            data = np.ndarray.tolist(R_3d[i, :, j])
            df = pd.DataFrame([data],columns=common)
            df[to_drop_ch] = 0
            df = df*1e-6
            df = df.reindex(columns=montage.ch_names)
            info = mne.create_info(ch_names=montage.ch_names,sfreq=10,ch_types='eeg')
            comp1 = mne.EvokedArray(df.to_numpy().T,info)
            comp1.set_montage(montage)
            comp1 = comp1.drop_channels(to_drop_ch)
            comp1.plot_topomap(times=[0],axes=axs[count],colorbar=False,show=False)
            ax[j, i].set_title(' ')
            ax[0, i].set_ylabel('Component ' + str(i))
            ax[j, 0].set_xlabel('Subject ' + str(j))
            ax[j, 0].xaxis.set_label_position('top')
            print(count)
            count += 1
    plt.show()
'''

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
        print(pvaf(X,W,140))

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

# Percentage variance accounted for
def pvaf(X, W, reduction_dim):
    # Reconstructing data set
    projection = np.dot(W, np.transpose(X))
    #print('projection:', np.shape(projection))
    pvaf = []
    for i in range(reduction_dim):
        pvaf.append(100-100*np.mean(np.var(np.transpose(X)-projection[i,:]))/np.mean(np.var(X)))
    print("pvaf: ", pvaf, 'pvaf_sum', np.sum(pvaf))
