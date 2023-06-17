import numpy as np
import mne
import os as os
#from ICA_dataImport import common, data_As, data_Vs, data_As, data_AVcs, data_AVics, data_AVc, data_A, data_V, data_AVic
import matplotlib.pyplot as plt
from our_group_ICA import  plotCumulativeExplainedVariances, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage, componentTimeseriesPlot, componentTimeseriesPlotIndividual, pvaf_new_ICA
from ICA_group_retry import backproj, whitening_matrix, estimated_mixing_matrices, R_pca, sorted

###############################################################################

# Import data from files made in ICA_dataImport.py
data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()

'''
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
'''

###############################################################################

# GA for all channels

np.random.seed(42)

num_subjects = 14  # Number of subjects 
num_samples = 54708  # Number of samples per subject # 13677*4
num_sources = 36  # Number of independent sources per subject # 36
num_mixtures = num_sources  # Number of observed mixtures per subject
n_components = 12

###############################################################################

data = np.zeros((num_subjects, num_sources, num_samples))

for i in range(num_subjects):
    data[i] = np.concatenate((data_A[i,:,:], data_V[i,:,:], data_As[i,:,:], data_Vs[i,:,:]), axis=1)

print('Shape of data: ', data.shape)

R = R_pca
G = whitening_matrix
A = estimated_mixing_matrices

print('Shape of R: ', R.shape)
print('Shape of G: ', G.shape)
print('Shape of A: ', A.shape)

ICA_comp = A[:, sorted[4]].reshape(168,1)

print('Shape of ICA_comp: ', ICA_comp.shape)

back_data = np.zeros((num_subjects, num_samples, num_subjects*n_components))

d_A = data[:, :, :97*141]
d_V = data[:, :, 97*141:2*97*141]
d_As = data[:, :, 2*97*141:3*97*141]
d_Vs = data[:, :, 3*97*141:]

print('Shape of d_A: ', d_A.shape)
print('Shape of d_V: ', d_V.shape)
print('Shape of d_As: ', d_As.shape)
print('Shape of d_Vs: ', d_Vs.shape)

# S = X * R * G * A
# X er for hver stimuli
# A er den enkelte udvalgte komponent

S_A = np.zeros((num_subjects, 13677, 1))
S_V = np.zeros((num_subjects, 13677, 1))
S_As = np.zeros((num_subjects, 13677, 1))
S_Vs = np.zeros((num_subjects, 13677, 1))

for i in range(num_subjects):
    S_A[i] = d_A[i,:,:].T @ R[i,:,:] @ G.T[i*n_components:(i+1)*n_components, :] @ ICA_comp
    S_V[i] = d_V[i,:,:].T @ R[i,:,:] @ G.T[i*n_components:(i+1)*n_components, :] @ ICA_comp
    S_As[i] = d_As[i,:,:].T @ R[i,:,:] @ G.T[i*n_components:(i+1)*n_components, :] @ ICA_comp
    S_Vs[i] = d_Vs[i,:,:].T @ R[i,:,:] @ G.T[i*n_components:(i+1)*n_components, :] @ ICA_comp

print('Shape of S_A: ', S_A.shape)
print('Shape of S_V: ', S_V.shape)
print('Shape of S_As: ', S_As.shape)
print('Shape of S_Vs: ', S_Vs.shape)

pdata_As = np.mean(S_As[0,:,:].reshape(97,141),axis=0)
pdata_Vs = np.mean(S_Vs[0,:,:].reshape(97,141),axis=0)
pdata_A = np.mean(S_A[0,:,:].reshape(97,141),axis=0)
pdata_V = np.mean(S_V[0,:,:].reshape(97,141),axis=0)

for i in range(1,14):
    pdata_As = np.vstack((pdata_As,np.mean(S_As[i,:,:].reshape(97,141),axis=0)))
    pdata_Vs = np.vstack((pdata_Vs,np.mean(S_Vs[i,:,:].reshape(97,141),axis=0)))
    pdata_A = np.vstack((pdata_A,np.mean(S_A[i,:,:].reshape(97,141),axis=0)))
    pdata_V = np.vstack((pdata_V,np.mean(S_V[i,:,:].reshape(97,141),axis=0)))

print('Shape of pdata_A: ', pdata_A.shape)
print('Shape of pdata_V: ', pdata_V.shape)
print('Shape of pdata_As: ', pdata_As.shape)
print('Shape of pdata_Vs: ', pdata_Vs.shape)

# Taking mean across subjects
p_As = np.mean(pdata_As,axis=0)
p_Vs = np.mean(pdata_Vs,axis=0)
p_A = np.mean(pdata_A,axis=0)
p_V = np.mean(pdata_V,axis=0)

x = np.arange(-0.1,1,step=1/128)
fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.axvline(x=0.1,color='r')
ax1.plot(x,p_As,color='k', label='A')
ax1.plot(x,p_Vs,color='k',linestyle='dashed', label='V')
#ax1.plot(x,pdata_AVcs,color='0.8', label = 'Congruent AV-V')
ax1.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax1.invert_yaxis()
ax1.set_title('Speech')
#ax2.axvline(x=0.1,color='r')
ax2.plot(x,p_A,color='k', label='A')
ax2.plot(x,p_V,color='k',linestyle='dashed', label='V')
#ax2.plot(x,pdata_AVc,color='0.8', label = 'Congruent AV-V')
ax2.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax2.invert_yaxis()
ax2.set_title('Non-speech')
ax1.legend(loc='upper right', fontsize = 8)
ax2.legend(loc='upper right', fontsize = 8)
plt.show()