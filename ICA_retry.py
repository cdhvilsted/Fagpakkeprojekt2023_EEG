################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import  plotCumulativeExplainedVariances, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage, componentTimeseriesPlot, componentTimeseriesPlotIndividual
from tqdm import tqdm
import time
import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import random

###############################################################################

# Import data from files made in ICA_dataImport.py
data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()
np.random.seed(42)
np.random.seed(42)


num_subjects = np.shape(data_A)[0]  # Number of subjects # 14
num_samples = np.shape(data_A)[2]  # Number of samples per subject # 13677
num_sources = np.shape(data_A)[1]  # Number of independent sources per subject # 36
num_mixtures = num_sources  # Number of observed mixtures per subject




# Perform PCA on the observed mixtures for each subject
S_pca = np.zeros((num_subjects, num_samples, num_sources))
A_pca = np.zeros((num_subjects, num_sources, num_sources))
X_pca = np.zeros((num_subjects, num_samples, num_sources))
mu_pca = np.zeros((num_subjects, num_sources))
for i in range(num_subjects):
    pca = PCA(n_components=None, whiten=False)
    X_pca[i] = pca.fit(data_A[i].T).transform(data_A[i].T)
    #S_pca[i] = pca.fit_transform(mixtures[i])
    A_pca[i] = pca.components_.T  # Get estimated mixing matrix
    S_pca[i] = np.dot(S_pca[i], A_pca[i].T)  # Estimate the sources
    mu_pca[i] = pca.mean_  # Get the mean of the mixtures
    assert np.allclose(data_A[i].T, np.dot(X_pca[i], A_pca[i].T)+mu_pca[i])


#concatenate the mixtures
X_pca = np.concatenate(X_pca, axis=1)

# make whitening pca
# Perform ICA on the PCA-transformed mixtures
ica = FastICA(n_components=None, whiten='arbitrary-variance', max_iter=100, tol=0.001)
estimated_sources = ica.fit_transform(X_pca)

#get whitening matrix
whitening_matrix = ica.whitening_


# get estimated mixing matrices
estimated_mixing_matrices = ica.mixing_
# get A
estimated_A = ica.components_


assert np.allclose(X_pca, np.dot(estimated_sources, estimated_mixing_matrices.T)+ica.mean_)
print("Success!")

# print all the shapes please
print(np.shape(estimated_sources), "estimated sources")
print(np.shape(estimated_mixing_matrices), "estimated mixing matrices")
print(np.shape(A_pca[0]), "A_pca")
print(np.shape(whitening_matrix), "whitening matrix")
print(np.shape(estimated_A), "estimated A")



# Common channels
common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']

import numpy as np
import matplotlib.pyplot as plt
from mne import create_info
from mne import EpochsArray
from mne.viz import plot_topomap

# Assuming you have a 3D array 'component_data' containing the independent components

# Create MNE info object
info = create_info(ch_names=common, sfreq=128, ch_types='eeg')
