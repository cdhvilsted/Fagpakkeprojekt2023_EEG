import mne
import sklearn as sk
import numpy as np
import random
import os




randomSeed = random.seed(1)

directory = os.path.dirname(os.path.abspath(__file__))+"/data_preproc"

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

epochs_list = []  # Create an empty list to store the epochs objects
# Define the list of filenames
Speech_files = [i + "_4adj.set" for i in Speech]

# Loop over the filenames and perform ICA on each file
for filename in Speech_files:
    path = directory +"/"+ str(filename)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path)
    events = mne.find_events(raw, stim_channel=common)
    event_id = ['Tagi_A','Tabi_A']
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=1.5, baseline=(None, 0), reject=dict(eeg=100e-6), preload=True)
    ica = mne.preprocessing.ICA(n_components=20, method='fastica')
    ica.fit(epochs)
    ica.plot_components()
    ica.exclude = [0, 1, 2]
    epochs_cleaned = ica.apply(epochs)

