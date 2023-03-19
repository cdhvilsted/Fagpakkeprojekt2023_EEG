from data_load_cec import *
import sklearn as sk
import mne
import numpy as np
import random
from t_testing_cec import *



from sklearn.decomposition import FastICA
print(all_datans)



randomSeed = random.seed(43)
path = directory+"/PP02_4adj.set"
epochs = mne.read_epochs_eeglab(path, montage_units='dm')
epochs = all_datans
print(epochs.shape)
# concatenate the epochs
data_matrix_s = np.concatenate((N1_As, N1_ics, N1_cs, P2_As, P2_ics, P2_cs))
data_matrix_ns = np.concatenate((N1_Ans, N1_icns, N1_cns, P2_Ans, P2_icns, P2_cns))
epochs = data_matrix_ns
# change epochs to raw
epochs = mne.io.RawArray(all_datans, mne.create_info(common, 512))


# Create MNE Epochs object

# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=28, method='fastica',random_state=randomSeed) # 14 components for each group
#ica.fit(mne.EpochsArray(data_matrix_s, mne.create_info(Speech, 128)))
# Reshape data_matrix_ns to have the shape (n_epochs, n_channels, n_samples)
data_matrix_ns = np.expand_dims(data_matrix_ns, axis=0)  # Add a new axis for epochs
#data_matrix_ns = np.transpose(data_matrix_ns, (0, 2, 1))  # Swap axes to get (n_epochs, n_samples, n_channels)


#change data_matrix_ns to epochs
"""
Make sure that the shape is (n_epochs, n_channels, n_samples) 
and that n_channels and n_samples are consistent with the in
-formation contained in the common variable that you use to cre
-ate the info argument in mne.create_info(common, 512).
"""

ica.fit(epochs)

# Plot the topographic maps of the independent components
ica.plot_components()




