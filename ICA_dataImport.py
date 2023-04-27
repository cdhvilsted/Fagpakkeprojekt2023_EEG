import mne
import numpy as np
import random
import os

randomSeed = random.seed(1)

# Set directory
directory = os.path.dirname(os.path.abspath(__file__))+"/EEGproj-main/EEGproj-main/data_preproc"

print("Working directory:  ", directory[:30],"/.../",directory[-57:], " ") # just printing the path to the data


# Test subjets

Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15",
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]

Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19",
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]

# Adding filename to "_4adj.set"
Speech_files = [ i + "_4adj.set" for i in Speech]
Non_speech_files = [i + "_4adj.set" for i in Non_speech]

# Common channels
common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']


Aud_event = ["Tagi_A", "Tabi_A"] #only auditive

Vis_event = ["Tagi_V","Tabi_V"] #only visual

# Importing data
aud1person = None
file = Speech_files[1]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm',verbose=False)
montage = raw.get_montage()
raw.pick_channels(common,verbose='WARNING')
raw = raw.crop(tmin=-0.1)
event1 = raw[Aud_event[0]] # tagi
event2 = raw[Aud_event[1]] # tabi
aud1person = mne.concatenate_epochs([event1, event2],verbose=False)
aud1person=aud1person.drop([i for i in range(130,len(aud1person))],verbose=False)
aud1person = aud1person.get_data()
aud1person = np.swapaxes(aud1person, 1,2).reshape(36,-1)


EEGdata = np.array(aud1person)
print("Dimensions of the data for 1 person:", np.shape(EEGdata))
for i in range(1,len(Speech_files)):
    file = Speech_files[i]
    path = directory + '/' + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm',verbose=False)
    raw.pick_channels(common,verbose=False)
    raw = raw.crop(tmin=-0.1)
    event1 = raw[Aud_event[0]] # tagi
    event2 = raw[Aud_event[1]] # tabi
    aud1person = mne.concatenate_epochs([event1, event2],verbose=False)
    aud1person=aud1person.drop([i for i in range(130,len(aud1person))],verbose=False)
    aud1person = aud1person.get_data()
    aud1person = np.swapaxes(aud1person, 1,2).reshape(36,-1)
    EEGdata = np.dstack((EEGdata,aud1person)) #hstack giver (trials, personer*antal kanaler, timesteps)


EEGdata = np.swapaxes(EEGdata,0,1)
EEGdata = np.swapaxes(EEGdata, 0,2)


print("Dimesions for data for all persons", np.shape(EEGdata))
