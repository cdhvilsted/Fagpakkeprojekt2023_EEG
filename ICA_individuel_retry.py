################# individuel ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import  plotCumulativeExplainedVariances, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage, componentTimeseriesPlot, componentTimeseriesPlotIndividual, pvaf_new_ICA
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

num_subjects = 1  # Number of subjects # 14
num_samples = np.shape(data_A)[2]  # Number of samples per subject # 13677
num_sources = np.shape(data_A)[1]  # Number of independent sources per subject # 36
num_mixtures = num_sources  # Number of observed mixtures per subject
subject_index = 0

###############################################################################

mixtures = np.concatenate((data_A[subject_index,:,:], data_V[subject_index,:,:], data_AVc[subject_index,:,:], data_AVic[subject_index,:,:]), axis=1)

print('Shape of data: ', mixtures.shape)

pca = PCA(n_components=36, whiten=False)
S_pca = pca.fit_transform(mixtures.T) # PCA wants (n_samples, n_features) thereby transposed
Rt_pca = pca.components_ # A is outputted transposed
X_pca = np.dot(S_pca, Rt_pca) + pca.mean_

# Reducing dimensionality
n_components = 12
X_reduced = np.dot(mixtures.T, Rt_pca[:n_components,:].T)
print('Shape of X_reduced: ', X_reduced.shape)

# Reality check
assert np.allclose(mixtures.T, X_pca)
print('Shape of X_pca: ', X_pca.shape)

ica = FastICA(n_components=None, whiten='arbitraty-variance', max_iter=400, tol=0.0001)
estimated_sources = ica.fit_transform(X_reduced)
whitening_matrix = ica.whitening_ # whitening matrix G
estimated_mixing_matrices = ica.mixing_ # estimated mixing matrices

sorted = pvaf_new_ICA(S = estimated_sources, X = X_reduced, A = estimated_mixing_matrices, reduction_dim = n_components, loop_range = n_components)

# assert for pca in ICA
assert np.allclose(X_reduced, np.dot(estimated_sources, estimated_mixing_matrices.T)+ica.mean_)
print("Success!")

# print all the shapes please
print(np.shape(estimated_sources), "estimated sources")
print(np.shape(estimated_mixing_matrices), "estimated mixing matrices")
print(np.shape(Rt_pca), "Rt_pca")
print(np.shape(whitening_matrix), "whitening matrix")

#estimated_sources1 = np.dot(estimated_sources, estimated_mixing_matrices.T)
estimated_mixing_matrices1 = np.dot(Rt_pca[:12,:].T, estimated_mixing_matrices)
#print(np.shape(estimated_sources1), "estimated sources1")
print(np.shape(estimated_mixing_matrices1), "estimated mixing matrices1")

componentTimeseriesPlotIndividual(estimated_mixing_matrices1.T, estimated_sources, numberComponents=7, numberSubjects=1, plotTitle='individuel ICA', sorted=sorted)