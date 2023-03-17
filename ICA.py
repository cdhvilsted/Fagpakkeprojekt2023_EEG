from data_load_cec import *
import sklearn as sk
import mne
import numpy as np
import random

randomSeed = random.seed(10)

epochs = mne.read_epochs_eeglab(path, montage_units='dm')

# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=20, method='fastica')
ica.fit(epochs)

# Plot the topographic maps of the independent components
ica.plot_components()
