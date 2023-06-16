################# group ICA algoritm for the  EEG dataset #####################
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

num_subjects = 14  # Number of subjects # 14
num_samples = 54708  # Number of samples per subject # 13677*4
num_sources = 36  # Number of independent sources per subject # 36
num_mixtures = num_sources  # Number of observed mixtures per subject
n_components = 12

###############################################################################

mixtures = np.zeros((num_subjects, num_sources, num_samples))

for i in range(num_subjects):
    mixtures[i] = np.concatenate((data_A[i,:,:], data_V[i,:,:], data_AVc[i,:,:], data_AVic[i,:,:]), axis=1)

print('Shape of data: ', mixtures.shape)

n_components = 12
X_pca = np.zeros((num_subjects, num_samples, n_components))
R_pca = np.zeros((num_subjects, num_sources, n_components))
S_pca = np.zeros((num_subjects, num_samples, n_components))
mu_pca = np.zeros((num_subjects, num_sources))
for i in range(num_subjects):
    pca = PCA(n_components=n_components, whiten=False)
    S_pca[i] = pca.fit(mixtures[i].T).transform(mixtures[i].T)
    #S_pca[i] = pca.fit_transform(mixtures[i])
    R_pca[i] = pca.components_.T  # Get estimated mixing matrix
    X_pca[i] = np.dot(mixtures[i].T, R_pca[i,:,:12]) + pca.mean_[:12]  # Estimate the sources
    mu_pca[i] = pca.mean_  # Get the mean of the mixtures
    #assert np.allclose(mixtures[i].T, np.dot(X_pca[i], R_pca[i].T)+mu_pca[i])

X_pca_red = X_pca

# Concatenate the mixtures
X_pca_red = np.concatenate(X_pca_red, axis=1)

print('Shape of X_pca_red: ', X_pca_red.shape)

# Perform ICA on the PCA-transformed mixtures
ica = FastICA(n_components=None, whiten='arbitrary-variance', max_iter=100, tol=0.001)
estimated_sources = ica.fit_transform(X_pca_red)

#get whitening matrix 
whitening_matrix = ica.whitening_
# get estimated mixing matrices
estimated_mixing_matrices = ica.mixing_

assert np.allclose(X_pca_red, np.dot(estimated_sources, estimated_mixing_matrices.T)+ica.mean_)
print("Success!")

# print all the shapes please
print(np.shape(estimated_sources), "estimated sources")
print(np.shape(estimated_mixing_matrices), "estimated mixing matrices")
print(np.shape(R_pca), "R_pca")
print(np.shape(whitening_matrix), "whitening matrix")

# Backprojection
backproj = np.zeros((num_subjects, n_components*num_subjects, num_sources))

# plot the components
for i in range(num_subjects):
    G_i = whitening_matrix.T[i*n_components:(i+1)*n_components, :]
    R_i = R_pca[i]
    A = estimated_mixing_matrices

    backproj[i] = np.transpose(np.dot(np.dot(R_i, G_i), A))

print('Shape of backproj: ', backproj.shape)

#s = np.arange(0, num_subjects*num_sources)
sorted = pvaf_new_ICA(S = estimated_sources, X = X_pca_red, A = estimated_mixing_matrices, reduction_dim = n_components, loop_range = n_components*num_subjects)

#componentTimeseriesPlot(backproj, estimated_sources, numberComponents=7, numberSubjects=14, plotTitle='group ICA', sorted=sorted)
#timeSeriesPlotICA(estimated_sources, sorted[1], plotTitle='Component 3')

fig, ax = plt.subplots(1, n_components, figsize=(15, 5))
for i in range(n_components):
    U3 = estimated_sources[:, i]
    U3 = U3.reshape(97*4, 141)
    U3 = np.mean(U3, axis=0)
    #plt.gca().invert_yaxis()
    #plt.plot(np.arange(-0.1, 1, step=1 / 128), U3)
    ax[0,i].set_title('Component ' + str(i))

plt.show()


