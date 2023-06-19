import numpy as np
import mne
import os as os
#from ICA_dataImport import common, data_As, data_Vs, data_As, data_AVcs, data_AVics, data_AVc, data_A, data_V, data_AVic
import matplotlib.pyplot as plt
from our_group_ICA import  plotCumulativeExplainedVariances, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage, componentTimeseriesPlot, componentTimeseriesPlotIndividual, pvaf_new_ICA
from ICA_group_retry import backproj, whitening_matrix, estimated_mixing_matrices, R_pca, sorted, data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics, num_sources, n_components, num_subjects, n_components

from sklearn.decomposition import PCA, FastICA

###############################################################################

# Import data from files made in ICA_dataImport.py
data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics

# GA for Cz channel
#plot data for speech files
pdata_As = np.mean(data_As[0,12,:].reshape(97,141),axis=0)
pdata_Vs = np.mean(data_Vs[0,12,:].reshape(97,141),axis=0)
pdata_AVcs = np.mean(data_AVcs[0,12,:].reshape(97,141),axis=0)
pdata_AVics = np.mean(data_AVics[0,12,:].reshape(97,141),axis=0)

#plot data for non-speech files
pdata_AVc = np.mean(data_AVc[0,12,:].reshape(97,141),axis=0)
pdata_A = np.mean(data_A[0,12,:].reshape(97,141),axis=0)
pdata_V = np.mean(data_V[0,12,:].reshape(97,141),axis=0)
pdata_AVic = np.mean(data_AVic[0,12,:].reshape(97,141),axis=0)
for i in range(1,14):
    pdata_As = np.vstack((pdata_As,np.mean(data_As[i,12,:].reshape(97,141),axis=0)))
    pdata_Vs = np.vstack((pdata_Vs,np.mean(data_Vs[i,12,:].reshape(97,141),axis=0)))
    pdata_AVcs = np.vstack((pdata_AVcs,np.mean(data_AVcs[i,12,:].reshape(97,141),axis=0)))
    pdata_AVics = np.vstack((pdata_AVics,np.mean(data_AVics[i,12,:].reshape(97,141),axis=0)))
    pdata_AVc = np.vstack((pdata_AVc,np.mean(data_AVc[i,12,:].reshape(97,141),axis=0)))
    pdata_A = np.vstack((pdata_A,np.mean(data_A[i,12,:].reshape(97,141),axis=0)))
    pdata_V = np.vstack((pdata_V,np.mean(data_V[i,12,:].reshape(97,141),axis=0)))
    pdata_AVic = np.vstack((pdata_AVic,np.mean(data_AVic[i,12,:].reshape(97,141),axis=0)))

print(pdata_As.shape)
print(np.mean(pdata_As,axis=0).shape)
pdata_As = np.mean(pdata_As,axis=0)
pdata_Vs = np.mean(pdata_Vs,axis=0)
pdata_AVcs = np.mean(pdata_AVcs,axis=0) - pdata_Vs
pdata_AVics = np.mean(pdata_AVics,axis=0) - pdata_Vs

pdata_A = np.mean(pdata_A,axis=0)
pdata_V = np.mean(pdata_V,axis=0)
pdata_AVic = np.mean(pdata_AVic,axis=0) - pdata_V
pdata_AVc = np.mean(pdata_AVc,axis=0) - pdata_V

x = np.arange(-0.1,1,step=1/128)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,pdata_As,color='k', label='A')
ax1.plot(x,pdata_AVcs,color='k',linestyle='dashed', label='Congruent AV-V')
ax1.plot(x,pdata_AVics,color='0.8', label = 'In-congruent AV-V')
#ax1.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax1.invert_yaxis()
ax1.set_title('Speech')
ax2.plot(x,pdata_A,color='k', label='A')
ax2.plot(x,pdata_AVc,color='k',linestyle='dashed', label='Congruent AV-V')
ax2.plot(x,pdata_AVic,color='0.8', label = 'In-congruent AV-V')
#ax2.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax2.invert_yaxis()
ax2.set_title('Non-speech')
ax1.legend(loc='upper right', fontsize = 8)
ax2.legend(loc='upper right', fontsize = 8)
#fig.suptitle('Grand average for "Cz" channel')

###############################################################################

# GA for all channels

np.random.seed(42)

num_subjects = num_subjects  # Number of subjects 
num_samples = 13677  # Number of samples per subject
num_sources = num_sources  # Number of independent sources per subject # 36
num_mixtures = num_sources  # Number of observed mixtures per subject
n_components = n_components
num_data = 8

data = np.zeros((num_data, num_subjects, num_sources, num_samples))

for i in range(num_subjects):
    data[0,i] = data_A[i,:,:] - data_A[i,:,:].mean(axis=1).reshape(36,1)
    data[1,i] = data_V[i,:,:] - data_V[i,:,:].mean(axis=1).reshape(36,1)
    data[2,i] = data_AVc[i,:,:] - data_AVc[i,:,:].mean(axis=1).reshape(36,1)
    data[3,i] = data_AVic[i,:,:] - data_AVic[i,:,:].mean(axis=1).reshape(36,1)
    data[4,i] = data_As[i,:,:] - data_As[i,:,:].mean(axis=1).reshape(36,1)
    data[5,i] = data_Vs[i,:,:] - data_Vs[i,:,:].mean(axis=1).reshape(36,1)
    data[6,i] = data_AVcs[i,:,:] - data_AVcs[i,:,:].mean(axis=1).reshape(36,1)
    data[7,i] = data_AVics[i,:,:] - data_AVics[i,:,:].mean(axis=1).reshape(36,1)

print('Shape of data: ', data.shape)


A = estimated_mixing_matrices
W = np.linalg.inv(A) #tried using the transposed version of A, but it didn't work

comp_chosen = 4

#ICA_comp2 = A[:, sorted[comp_chosen]].reshape(n_components*num_subjects,1)
ICA_comp2 = W[:, sorted[comp_chosen]].reshape(n_components*num_subjects,1) 


X_pca_liste = np.zeros((num_data,num_subjects, n_components, num_samples))
R_pca_liste = np.zeros((num_data,num_subjects, num_sources, n_components))
S_pca_liste = np.zeros((num_data,num_subjects, num_samples, num_sources))
mu_pca_liste = np.zeros((num_data,num_subjects, num_sources))

for j in range(num_data):
    for i in range(num_subjects): 
        pca = PCA(n_components=None, whiten=False)
        S_pca_liste[j,i,:,:] = pca.fit(data[j,i].T).transform(data[j,i].T)
        R1 = pca.components_.T
        R_pca_liste[j,i,:,:] = R1[:,:n_components]# Get estimated mixing matrix
        X_pca_liste[j,i,:,:] = np.transpose(np.dot(data[j,i].T, R_pca_liste[j,i,:,:]))  # Estimate the sources (maybe add mean)
        mu_pca_liste[j,i,:] = pca.mean_ # Get the mean of the mixtures


# Dividing data into specific stimuli
d_A = X_pca_liste[0,:, :, :]
d_V = X_pca_liste[1,:, :, :]
d_AVc = X_pca_liste[2, :, :, :]
d_AVic = X_pca_liste[3, :, :, :]

d_As = X_pca_liste[4,:, :, :]
d_Vs = X_pca_liste[5,:, :, :]
d_AVcs = X_pca_liste[6, :, :, :]
d_AVics = X_pca_liste[7, :, :, :]

# S = A_comp * X_reduced
# X er for hver stimuli
# A er den enkelte udvalgte komponent

S_A = np.zeros((num_subjects, 13677, 1))
S_V = np.zeros((num_subjects, 13677, 1))
S_AVc = np.zeros((num_subjects, 13677, 1))
S_AVic = np.zeros((num_subjects, 13677, 1))

S_As = np.zeros((num_subjects, 13677, 1))
S_Vs = np.zeros((num_subjects, 13677, 1))
S_AVcs = np.zeros((num_subjects, 13677, 1))
S_AVics = np.zeros((num_subjects, 13677, 1))

for i in range(num_subjects):
    S_A[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_A[i,:,:]).T
    S_V[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_V[i,:,:]).T
    S_AVc[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_AVc[i,:,:]).T
    S_AVic[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_AVic[i,:,:]).T
    S_As[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_As[i,:,:]).T
    S_Vs[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_Vs[i,:,:]).T
    S_AVcs[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_AVcs[i,:,:]).T
    S_AVics[i] = (ICA_comp2.T[:, i*n_components:(i+1)*n_components] @ d_AVics[i,:,:]).T

# Taking mean across epochs for each subject
pdata_As = np.mean(S_As[0,:,:].reshape(97,141),axis=0)
pdata_Vs = np.mean(S_Vs[0,:,:].reshape(97,141),axis=0)
pdata_AVcs = np.mean(S_AVcs[0,:,:].reshape(97,141),axis=0)
pdata_AVics = np.mean(S_AVics[0,:,:].reshape(97,141),axis=0)

pdata_A = np.mean(S_A[0,:,:].reshape(97,141),axis=0)
pdata_V = np.mean(S_V[0,:,:].reshape(97,141),axis=0)
pdata_AVc = np.mean(S_AVc[0,:,:].reshape(97,141),axis=0)
pdata_AVic = np.mean(S_AVic[0,:,:].reshape(97,141),axis=0)

for i in range(1,num_subjects):
    pdata_As = np.vstack((pdata_As,np.mean(S_As[i,:,:].reshape(97,141),axis=0)))
    pdata_Vs = np.vstack((pdata_Vs,np.mean(S_Vs[i,:,:].reshape(97,141),axis=0)))
    pdata_AVcs = np.vstack((pdata_AVcs,np.mean(S_AVcs[i,:,:].reshape(97,141),axis=0)))
    pdata_AVics = np.vstack((pdata_AVics,np.mean(S_AVics[i,:,:].reshape(97,141),axis=0)))
    pdata_A = np.vstack((pdata_A,np.mean(S_A[i,:,:].reshape(97,141),axis=0)))
    pdata_V = np.vstack((pdata_V,np.mean(S_V[i,:,:].reshape(97,141),axis=0)))
    pdata_AVc = np.vstack((pdata_AVc,np.mean(S_AVc[i,:,:].reshape(97,141),axis=0)))
    pdata_AVic = np.vstack((pdata_AVic,np.mean(S_AVic[i,:,:].reshape(97,141),axis=0)))

# Taking mean across subjects
p_As = np.mean(pdata_As,axis=0)
p_Vs = np.mean(pdata_Vs,axis=0)
p_AVcs = np.mean(pdata_AVcs,axis=0) - p_Vs
p_AVics = np.mean(pdata_AVics,axis=0) - p_Vs
p_A = np.mean(pdata_A,axis=0)
p_V = np.mean(pdata_V,axis=0)
p_AVc = np.mean(pdata_AVc,axis=0) - p_V
p_AVic = np.mean(pdata_AVic,axis=0) - p_V

# Plotting grand average
x = np.arange(-0.1,1,step=1/128)
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.plot(x,p_As,color='k', label='A')
ax1.plot(x,p_AVcs,color='k',linestyle='dashed', label='Congruent AV-V')
ax1.plot(x,p_AVics,color='0.8', label = 'In-congruent AV-V')
#ax1.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax1.invert_yaxis()
ax1.set_title('Speech')
ax2.plot(x,p_A,color='k', label='A')
ax2.plot(x,p_AVc,color='k',linestyle='dashed', label='Congruent AV-V')
ax2.plot(x,p_AVic,color='0.8', label = 'In-congruent AV-V')
#ax2.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax2.invert_yaxis()
ax2.set_title('Non-speech')
ax1.legend(loc='upper right', fontsize = 8)
ax2.legend(loc='upper right', fontsize = 8)
#fig.suptitle('Grand average for "synthetic" channel')

plt.show()