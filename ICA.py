from data_load_cec import *
import sklearn as sk
import mne
import numpy as np
import random


randomSeed = random.seed(1)

directory = os.path.dirname(os.path.abspath(__file__))+"/EEGproj-main/EEGproj-main/data_preproc"

print("Working directory:  ", directory[:30],"/.../",directory[-57:], " ") # just printing the path to the data

Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15",
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]
Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19",
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]
Speech_files = [ i + "_4adj.set" for i in Speech]
Non_speech_files = [i + "_4adj.set" for i in Non_speech]
common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']


aud1person = None
file = Speech_files[2]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
raw = raw.crop(tmin=-0.1)

event1 = raw[Aud_event[0]] # tabi
event2 = raw[Aud_event[1]] # tagi (maybe)

aud1person = mne.concatenate_epochs([event1, event2])
print("shape:",len(aud1person[0]))


# Create MNE Epochs object

# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=28, method='fastica',random_state=randomSeed ) # 14 components for each group
#ica.fit(mne.EpochsArray(data_matrix_s, mne.create_info(Speech, 128)))


#change data_matrix_ns to epochs
"""
Make sure that the shape is (n_epochs, n_channels, n_samples) 
and that n_channels and n_samples are consistent with the in
-formation contained in the common variable that you use to cre
-ate the info argument in mne.create_info(common, 512).
"""

ica.fit(aud1person)
# make a for loop for i in range
print([ica.get_explained_variance_ratio(aud1person, components=i, ch_type='eeg') for i in range(0,28)])


# Plot the topographic maps of the independent components
ica.plot_components()


