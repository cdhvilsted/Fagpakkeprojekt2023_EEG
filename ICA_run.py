################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA, pvaf
from tqdm import tqdm
import time

###############################################################################

# Import data
from ICA_dataImport import EEGdata, montage, common

# First PCA (whitening)
print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
print("")
print("# This is the first PCA: #")
print("")

reduceDimensions = 36
print("Dimensions chosen: ", 18330)
print("")
print('EEG', EEGdata[0].shape)
X_pca1 = np.array([])
R = np.array([])
print("doing PCA on each subject")
for i in range(0, 14):
   U, S, V, reduced_X, rho = PCA(EEGdata[i].T, reduceDimensions, plot=False)
   if len(X_pca1) == 0:
        X_pca1 = reduced_X
        R = np.transpose(V[:,:reduceDimensions])
        R_3d = np.transpose(V[:,:reduceDimensions])
        U_3d = np.transpose(U[:,:reduceDimensions])

   else:
       X_pca1 = np.hstack((reduced_X,X_pca1))
       R = np.vstack((np.transpose(V[:,:reduceDimensions]),R))
       R_3d = np.dstack((R_3d, np.transpose(V[:,:reduceDimensions])))
       U_3d = np.dstack((U_3d, np.transpose(U[:,:reduceDimensions])))

#print('reduced', (R[:,:,0]@EEGdata[0]).shape)

print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape, 'R:', R.shape, 'R_3d:', R_3d.shape)
X_concat = X_pca1.T
print("X_concat shape: ", X_concat.shape)

biosemi_montage = mne.channels.make_standard_montage('standard_1020',head_size=0.15)
print(montage.ch_names)
to_drop_ch = list(set(montage.ch_names)-set(common))
print(len(to_drop_ch))

'''
fig, ax = plt.subplots(4,14, figsize=(10,7))
#plt.subplots_adjust(hspace=0.5)
axs = ax.ravel()
count = 0
for j in range(14):
    for i in range(4):
        # make back_Y a list
        data = np.ndarray.tolist(R_3d[i,:,j])
        df = pd.DataFrame([data],columns=common)
        df[to_drop_ch] = 0
        #df = df*1e-6
        df = df.reindex(columns=montage.ch_names)
        info = mne.create_info(ch_names=montage.ch_names,sfreq=10,ch_types='eeg')
        comp1 = mne.EvokedArray(df.to_numpy().T,info)
        comp1.set_montage(montage)
        comp1 = comp1.drop_channels(to_drop_ch)
        comp1.plot_topomap(times=[0],axes=axs[count],colorbar=False,show=False)
        #ax[i, j].set_title(' ')
        #ax[j, 0].set_xlabel('Component ' + str(i))
        #ax[0, i].set_ylabel('Subject ' + str(j))
        #ax[j, 0].xaxis.set_label_position('top')
        print(count)
        count += 1
#comp1.plot()
#common1 =[i for i in biosemi_montage if i in common]
#print(np.shape(comp1))
#comp1 = mne.Epochs(comp1)
#print(type(comp1))
#print(comp1)
#comp1.plot_topomap(times=[0],sphere='eeglab')
plt.show()


U3 = U_3d[0,:,2]
print('U3:', U3.shape)
#S0 = S[sorted[-1]]
U3 = U3.reshape(141,130)
U3 = np.mean(U3, axis=1)
plt.gca().invert_yaxis()
plt.plot(np.arange(-0.1,1,step=1/128),U3)
plt.show()'''


print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# Second PCA
print("")
print("# This is the second PCA: #")
print("")

U, S, V, reduced_X, rho = PCA(X_concat.T, reduced_dim = 140, plot=False)
G = V #changed from U to V
print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape)
X_whithen = reduced_X
print("X shape: ", X_whithen.shape)

fig, ax = plt.subplots(4,14, figsize=(10,7))
#plt.subplots_adjust(hspace=0.5)
axs = ax.ravel()
count = 0
for j in range(14):
    for i in range(4):
        # make back_Y a list
        data = np.ndarray.tolist(G[i,:,j])
        df = pd.DataFrame([data],columns=common)
        df[to_drop_ch] = 0
        #df = df*1e-6
        df = df.reindex(columns=montage.ch_names)
        info = mne.create_info(ch_names=montage.ch_names,sfreq=10,ch_types='eeg')
        comp1 = mne.EvokedArray(df.to_numpy().T,info)
        comp1.set_montage(montage)
        comp1 = comp1.drop_channels(to_drop_ch)
        comp1.plot_topomap(times=[0],axes=axs[count],colorbar=False,show=False)
        #ax[i, j].set_title(' ')
        #ax[j, 0].set_xlabel('Component ' + str(i))
        #ax[0, i].set_ylabel('Subject ' + str(j))
        #ax[j, 0].xaxis.set_label_position('top')
        print(count)
        count += 1
plt.show()

print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# ICA
print("# This is the ICA step: #")
print("")

S, A, W, sorted = ICA(X_whithen.T, R, G, "fastICA") #X needs shape (n_samples, n_features)


print("S shape: ", S.shape, "     A shape: ", A.shape, "     W shape: ", W.shape)

print("")

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

#try to plot ESP from S
S = S.T
S0 = S[sorted[-1]]
S0 = S0.reshape(130,141)
S0 = np.mean(S0, axis=0)
plt.plot(np.arange(-0.1,1,step=1/128),S0)
plt.show()


back_Y = np.zeros((14,140,36))

print(np.transpose(R_3d[:,:,0]).shape, 'should be 36,10')
#print(np.transpose(G[:,10*(0):10*(0+1)]).shape, 'should be 10, 140')
print(G[10*(0):10*(0+1),:].shape, 'should be 10, 140')

print(np.linalg.pinv(W).shape, 'should be 140, 140')
print((np.transpose(R_3d[:,:,0]) @ np.transpose(G[:,10*(0):10*(0+1)]) @ np.linalg.pinv(W)).shape)
for i in range(14):
    back_Y[i,:,:] = np.transpose(np.transpose(R_3d[:,:,i]) @ G[10*(0):10*(0+1),:] @ np.linalg.pinv(W))

'''
for i in range(14):
    for j in range(10):
        #back_Y[i, j, :] = np.matmul(A @ S[:, i].reshape((-1, 1)), np.ones((1, 36)))
        back_Y[i, j, :] = np.transpose(R_3d[:,:,j]) @ np.transpose(G[:,10*(i):10*(i+1)]) @ np.linalg.pinv(W)
'''
print(np.shape(back_Y))

# data for hver forsøgsperson kommer af at gange mixing matrix med source for hver forsøgsperson fx X0 = A[0,0,:] @ S[0,:]

# sorter efter varians i hver component
#print(A[0,0,:])
#var_S = np.var(S, axis=1)
#print(np.argsort(var_S)) # sort index from smallest to highest variance


biosemi_montage = mne.channels.make_standard_montage('standard_1020',head_size=0.15)
print(montage.ch_names)
to_drop_ch = list(set(montage.ch_names)-set(common))
print(len(to_drop_ch))


fig, ax = plt.subplots(14,10, figsize=(15,12))
#plt.subplots_adjust(hspace=0.5)
print(len(common))
axs = ax.ravel()
count = 0
for j in range(14):
    for i in range(10):
        # make back_Y a list
        data = np.ndarray.tolist(back_Y[j,sorted[-(i+1)]])
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
#comp1.plot()
#common1 =[i for i in biosemi_montage if i in common]
#print(np.shape(comp1))
#comp1 = mne.Epochs(comp1)
#print(type(comp1))
#print(comp1)
#comp1.plot_topomap(times=[0],sphere='eeglab')
plt.show()







