################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA
###############################################################################

# Import data
from ICA_dataImport import EEGdata, montage, common

# First PCA (whitening)
print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
print("")
print("# This is the first PCA: #")
print("")


reduceDimensions = 10
print("Dimensions chosen: ", 18330)
print("")

X_pca1 = np.array([])
print("doing PCA on each subject")
for i in range(0, 14):
   U, S, V, reduced_X, rho = PCA(EEGdata[i].T, reduceDimensions, plot=False)
   if len(X_pca1) == 0:
        X_pca1 = reduced_X
   else: X_pca1 = np.hstack((reduced_X,X_pca1))


print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape)
R = X_pca1.T
print("R shape: ", R.shape)



print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# Second PCA
print("")
print("# This is the second PCA: #")
print("")

U, S, V, reduced_X, rho = PCA(R, reduced_dim = 140, plot=False)
print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape)
G = reduced_X
print("G shape: ", G.shape)




print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# ICA
print("# This is the ICA step: #")
print("")

S = ICA(G)
print("")

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")
print("# Reconstruction: #")
print("")
print("S shape: ", S.shape)
print(S)

biosemi_montage = mne.channels.make_standard_montage('standard_1020',head_size=0.15)
to_drop_ch = list(set(montage.ch_names)-set(common))


fig, ax = plt.subplots(14,10, figsize=(15,12))


axs = ax.ravel()
count = 0
for j in range(14):
    for i in range(10):
        data = np.ndarray.tolist(np.array(S[j,:]))
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
        ax[0, i].set_ylabel('Subject ' + str(i))
        ax[j, 0].set_xlabel('Component ' + str(j))
        ax[j, 0].xaxis.set_label_position('top')
        print(count)
        count += 1

plt.show()








