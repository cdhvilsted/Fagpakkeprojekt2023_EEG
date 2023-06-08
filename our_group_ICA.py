###################### group ICA algoritm for EEG data ########################
###############################################################################

# packages
import os

import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
from picard import picard
#import fast ICA
from sklearn.decomposition import FastICA
from tqdm import tqdm
import time
#from ICA_dataImport import common, montage
import pandas as pd

###############################################################################


# Set directory
directory = os.path.dirname(os.path.abspath(__file__))+"/EEGproj-main/EEGproj-main/data_preproc"

print("Working directory:  ", directory[:30],"/.../",directory[-57:], " ") # just printing the path to the data


# Test subjets

Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15",
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]

Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19",
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]

# Adding filename to "_4adj.set"
Speech_files = [ i + "_4adj.set" for i in Speech]
Non_speech_files = [i + "_4adj.set" for i in Non_speech]

# Common channels
common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']


Aud_event = ["Tagi_A", "Tabi_A"] #only auditive

Vis_event = ["Tagi_V","Tabi_V"] #only visual

AV_c_event = ["Tagi_A_Tagi_V","Tabi_A_Tabi_V"] #congruent audivisual

AV_ic_event = ["Tagi_A_Tabi_V","Tabi_A_Tagi_V"] # incongruent audivisual

numofepochs = 97 #number of epochs per condition
timestep = 141 #number of timepoints per epech

#loading one person to get the montage
aud1person = None
file = Speech_files[0]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm',verbose=False)
montage = raw.get_montage()
print(" montage: ",montage)




def loadData():
    saves = ["data_A", "data_V", "data_AVc", "data_AVic", "data_As", "data_Vs", "data_AVcs", "data_AVics"]
    total_files = len(saves)
    loaded_files = []

    with tqdm(total=total_files, desc="Loading data", unit="file", ncols=80) as pbar:
        for save in saves:
            filename = save + ".txt"
            loaded_data_2d = np.loadtxt(filename, delimiter=',')
            loaded_data_3d = loaded_data_2d.reshape(
                loaded_data_2d.shape[0], loaded_data_2d.shape[1] // 13677, 13677)
            globals()["load_3d_" + save] = loaded_data_3d

            # Add the loaded file to the list and update the loading bar
            loaded_files.append(filename)
            pbar.set_postfix(current_file=filename)
            pbar.update(1)

    print("All files loaded successfully!")

    # Return all data
    return (
        load_3d_data_A, load_3d_data_V, load_3d_data_AVc, load_3d_data_AVic,
        load_3d_data_As, load_3d_data_Vs, load_3d_data_AVcs, load_3d_data_AVics
    )



def plotCumulativeExplainedVariances(rho, threshold = 0.95):
    rho = np.diagonal(rho)
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

    '''
    # checking if error is small enough to ignore
    if np.mean(abs(U @ S @ Vt - X)) < 0.0000000000000002:
        print('SVD works correctly')
        print ("error = ",np.mean(abs(U @ S @ Vt - X)))
    else:
        print('SVD does not work, error seems large')
        print ("error = ",np.mean(abs(U @ S @ Vt - X)))
    '''
    
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

def componentPlot(R, numberComponents, numberSubjects, plotTitle):
    biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    to_drop_ch = list(set(montage.ch_names) - set(common))
    fig, ax = plt.subplots(numberComponents, numberSubjects, figsize=(20, 7))
    axs = ax.ravel()
    pbar = tqdm(total=numberComponents * numberSubjects)  # Initialize the progress bar
    count = 0
    minR = round(np.min(R),3)
    #print('-------------------')
    #print(np.min(R))
    #print(np.max(R))
    #print('-------------------')
    maxR = round(np.max(R),3)
    maxvalue = np.max([abs(minR), abs(maxR)])
    #cnorm = TwoSlopeNorm(vmin=-maxvalue, vcenter=0, vmax=maxvalue)
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
            #norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            comp1 = comp1.drop_channels(to_drop_ch)
            im = comp1.plot_topomap(times=[0], axes=axs[count], colorbar=False, cmap='RdBu_r', show=False, sphere=0.12, vlim=(-maxvalue*1e6,maxvalue*1e6))
            st = fig.suptitle(plotTitle, fontsize="x-large")
            pbar.update(1)  # Update the progress bar
            ax[i, j].set_title(' ')
            ax[0, j].set_xlabel('Subject ' + str(j))
            ax[i, 0].set_ylabel('Component ' + str(i))
            ax[0, j].xaxis.set_label_position('top')
            count += 1
    
    
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r'), ax=axs.ravel().tolist(),fraction=0.047*7/20)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([-maxvalue, 0, maxvalue])
    plt.show()


def timeSeriesPlot(U_3d, component, subject, plotTitle):
    U3 = U_3d[component, :, subject]
    U3 = U3.reshape(97, 141)
    U3 = np.mean(U3, axis=0)
    plt.gca().invert_yaxis()
    plt.plot(np.arange(-0.1, 1, step=1 / 128), U3)
    plt.title(plotTitle)
    plt.show()

def timeSeriesPlotICA(U_3d, component, plotTitle):
    U3 = U_3d[:, component]
    U3 = U3.reshape(97, 141)
    U3 = np.mean(U3, axis=0)
    plt.gca().invert_yaxis()
    plt.plot(np.arange(-0.1, 1, step=1 / 128), U3)
    plt.title(plotTitle)
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

def ICA(X, R, G, typeICA, reduced_dim):
    # this function takes a matrix and returns the ICA of the matrix
    if typeICA == "fastICA":
        # fastICA
        transform_ICA = sklearn.decomposition.FastICA(n_components = None,  whiten=False, fun='logcosh', max_iter=200, tol=0.0001, random_state=2) #added random state to make it reproducible

        # fit the model
        S = transform_ICA.fit_transform(X)
        A = transform_ICA.mixing_

        W = np.linalg.pinv(A)
        W_com = transform_ICA.components_
    
        # pvaf
        #print(pvaf(X,W_com, G, R, reduction_dim=reduced_dim))
        sorted = pvaf_source(S)
        

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
def pvaf(X, W, G, R, reduction_dim):
    # Reconstructing data set
    projection = np.dot(X, np.linalg.pinv(W))
    pvaf = []
    for i in range(reduction_dim):
        pvaf.append(100-100*np.mean(np.var(X-projection[i,:], axis=0))/np.mean(np.var(X, axis=0)))
    
    print("pvaf: ", pvaf, 'pvaf_sum', np.sum(pvaf))


def pvaf_source(S):
    print('S:', S.shape)
    explained = np.var(S, axis = 0)
    explained_ratio = explained / np.sum(explained)
    sorted = np.argsort(explained_ratio)[::-1]
    
    rho = np.cumsum(explained_ratio)
    numcomponents = np.where(rho > 0.95)[0][0]
    print("Number of components to keep 0.95 of data after ICA: ", numcomponents)
    # plotting explained variance in different ways
    #plt.bar(explained_ratio, height=1, width=0.2, edgecolor='black')
    #plt.hist(explained_ratio, bins=, edgecolor='black')
    #plt.plot(np.arange(0,49),explained_ratio[sorted], 'o', color='black')
    #plt.show()
    #print(max(explained_ratio), np.mean(explained_ratio), min(explained_ratio))
    return sorted