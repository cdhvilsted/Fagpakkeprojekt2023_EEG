from data_load_cec import *
import sklearn as sk
import mne
import numpy as np
import random
from t_testing_cec import *

randomSeed = random.seed(43)

epochs = mne.read_epochs_eeglab(path, montage_units='dm')

data_matrix_s = np.concatenate((N1_As, N1_ics, N1_cs, P2_As, P2_ics, P2_cs))
data_matrix_ns = np.concatenate((N1_Ans, N1_icns, N1_cns, P2_Ans, P2_icns, P2_cns))


# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=28, method='fastica',random_state=randomSeed) # 14 components for each group
#ica.fit(mne.EpochsArray(data_matrix_s, mne.create_info(Speech, 128)))
ica.fit(epochs)

# Plot the topographic maps of the independent components
ica.plot_components()

