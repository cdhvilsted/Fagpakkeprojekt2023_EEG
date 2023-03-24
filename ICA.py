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


aud1person = None
file = Speech_files[0]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
raw = raw.crop(tmin=-0.1)

# apply baseline
event1 = raw[Aud_event[0]].apply_baseline(baseline) # tabi
event2 = raw[Aud_event[1]].apply_baseline(baseline) # tagi (maybe)

aud1person = mne.concatenate_epochs([event1, event2])
print("shape:",len(aud1person[0]))

# Create MNE Epochs object

# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=28, method='fastica',random_state=randomSeed) # 14 components for each group
#ica.fit(mne.EpochsArray(data_matrix_s, mne.create_info(Speech, 128)))


#change data_matrix_ns to epochs
"""
Make sure that the shape is (n_epochs, n_channels, n_samples) 
and that n_channels and n_samples are consistent with the in
-formation contained in the common variable that you use to cre
-ate the info argument in mne.create_info(common, 512).
"""

ica.fit(aud1person)

# Plot the topographic maps of the independent components
ica.plot_components()




