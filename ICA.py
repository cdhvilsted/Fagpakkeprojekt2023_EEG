from data_load_cec import *
import sklearn as sk
import mne
import numpy as np
import random
from t_testing_cec import *



import mne
from sklearn.decomposition import FastICA

# Load the data for each subject
subject_data = []
for i in Speech_files:
    # Load the data for each subject, e.g. using MNE-Python
    data = mne.io.read_epochs_eeglab( directory +"/"+i,montage_units='dm')
    # Preprocess the data as needed
    data = data.filter(l_freq=1, h_freq=30)
    # Stack the data into a single long series
    stacked_data = np.hstack(data.get_data())
    subject_data.append(stacked_data)

# Create a matrix of all the stacked data
data_matrix = np.vstack(subject_data)

# Perform Group ICA using the FastICA algorithm
ica = FastICA(n_components=10, random_state=0)
ica.fit(data_matrix)

# Get the independent components
components = ica.components_

# Create an info object for the data
ch_names = common
ch_types = ['eeg'] * len(ch_names)
sfreq = 512
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create an MNE-Python RawArray object for each component
raw_arrays = []
for i in range(10):
    component_data = components[i]
    # Reshape the data into a 2D array with shape (n_channels, n_samples)
    component_data = component_data.reshape(len(ch_names), -1)
    raw = mne.io.RawArray(component_data, info)
    raw_arrays.append(raw)

# Plot each component using MNE-Python
for i in range(10):
    raw_arrays[i].plot(n_channels=len(ch_names), scalings='auto', title=f'ICA Component {i+1}')









randomSeed = random.seed(43)
path = directory+"/PP02_4adj.set"
epochs = mne.read_epochs_eeglab(path, montage_units='dm')

# concatenate the epochs
data_matrix_s = np.concatenate((N1_As, N1_ics, N1_cs, P2_As, P2_ics, P2_cs))
data_matrix_ns = np.concatenate((N1_Ans, N1_icns, N1_cns, P2_Ans, P2_icns, P2_cns))

#data_matrix_s = np.c((N1_As, N1_ics, N1_cs, P2_As, P2_ics, P2_cs))
#data_matrix_ns = np.vstack((N1_Ans, N1_icns, N1_cns, P2_Ans, P2_icns, P2_cns))

# Create MNE info object
n_channels = len(common)
n_samples = data_matrix_ns.shape[0]
print(len(list_files),n_channels,n_samples)
ch_names = ['ch{}'.format(i) for i in range(len(common))]
sfreq = 512  # Sampling rate
info = mne.create_info(ch_names=ch_names, sfreq=sfreq)

# Create MNE Epochs object

# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=28, method='fastica',random_state=randomSeed) # 14 components for each group
#ica.fit(mne.EpochsArray(data_matrix_s, mne.create_info(Speech, 128)))
# Reshape data_matrix_ns to have the shape (n_epochs, n_channels, n_samples)
data_matrix_ns = np.expand_dims(data_matrix_ns, axis=0)  # Add a new axis for epochs
#data_matrix_ns = np.transpose(data_matrix_ns, (0, 2, 1))  # Swap axes to get (n_epochs, n_samples, n_channels)

# Add new axis to create shape (n_epochs, n_samples, n_channels)
data_matrix_ns = data_matrix_ns[:, :, np.newaxis]
# Transpose the array to get shape (n_epochs, n_samples, n_channels)
data_matrix_ns = np.transpose(data_matrix_ns, (0, 1, 2))

epochs = mne.EpochsArray(data_matrix_ns, info)

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




