import sklearn as sk
import mne
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy

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

Aud_event = ["Tagi_A", "Tabi_A"] #only auditive

Vis_event = ["Tagi_V","Tabi_V"] #only visual

aud1person = None
file = Speech_files[1]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
montage = raw.get_montage()
raw.pick_channels(common)
raw = raw.crop(tmin=-0.1)
print(raw.info)
event1 = raw[Aud_event[0]] # tagi
event2 = raw[Aud_event[1]] # tabi
aud1person = mne.concatenate_epochs([event1, event2])
aud1person=aud1person.drop([i for i in range(130,len(aud1person))])
aud1person = aud1person.get_data()
aud1person = np.swapaxes(aud1person, 1,2).reshape(36,-1)


liste = np.array(mne.io.read_epochs_eeglab(directory + '/' + Speech_files[0].replace(" ",""), montage_units='dm')
                   .pick_channels(common)
                   .crop(tmin=-0.1)
                   [Aud_event])
for file in Speech_files[1:]:
    raw = mne.io.read_epochs_eeglab(directory + '/' + str(file).replace(" ",""), montage_units='dm')\
        .pick_channels(common)\
        .crop(tmin=-0.1)
    aud1person = mne.concatenate_epochs([raw[Aud_event[0]], raw[Aud_event[1]]])[:130]
    aud1person = np.swapaxes(aud1person.get_data(), 1, 2).reshape(36, -1)
    liste = np.dstack((liste, aud1person))
    print('Aud1person shape is:', aud1person.shape)
print(liste.shape)


liste = np.swapaxes(liste,0,1)
liste = np.swapaxes(liste, 0,2)

print("shape:",np.shape(liste[0]))
print(np.shape(liste))
