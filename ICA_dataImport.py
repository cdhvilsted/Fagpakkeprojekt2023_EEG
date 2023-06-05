import mne
import numpy as np
import random
import os
from tqdm import tqdm


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

AV_c_event = ["Tagi_A_Tagi_V","Tabi_A_Tabi_V"] #congruent audivisual

AV_ic_event = ["Tagi_A_Tabi_V","Tabi_A_Tagi_V"] # incongruent audivisual

numofepochs = 97 #number of epochs per condition
timestep = 141 #number of timepoints per epech

#loading one person to get the montage
aud1person = None
file = Speech_files[0]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm',verbose=False)
montage = raw.get_montage()



data_As = np.empty((14,36,numofepochs*timestep)) #auditive data for speech files
data_Vs = np.empty((14,36,numofepochs*timestep)) #visual data for speech files
data_AVcs = np.empty((14,36,numofepochs*timestep)) #congruent audivisual data for speech files
data_AVics = np.empty((14,36,numofepochs*timestep)) #incongruent audivisual data for speech files

data_A = np.empty((14,36,numofepochs*timestep)) #auditive data for non-speech files
data_V = np.empty((14,36,numofepochs*timestep)) #visual data for non-speech files
data_AVc = np.empty((14,36,numofepochs*timestep)) #congruent audivisual data for non-speech files
data_AVic = np.empty((14,36,numofepochs*timestep)) #incongruent audivisual data for non-speech files

counter2 = 0
print('-----------------------------------------')
print('Processing speech files...')
pbar = tqdm(total=14 * 36 * 4)  # Initialize the progress bar
for file in Speech_files:
    path = directory+"/ " + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm',verbose=False)
    raw = raw.crop(tmin=-0.1)
    counter = 0
    for chan in common:
        #auditive data
        event1 = raw[Aud_event[0]] # tagi
        event2 = raw[Aud_event[1]] # tabi
        aud1person = mne.concatenate_epochs([event1, event2],verbose=False)
        aud1person=aud1person.drop([i for i in range(numofepochs,len(aud1person))],verbose=False)
        aud1person = aud1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        aud1person = aud1person.get_data(picks=chan).flatten()
        data_As[counter2, counter] = aud1person
        pbar.update(1)  # Update the progress bar

        #visual data
        event1 = raw[Vis_event[0]] 
        event2 = raw[Vis_event[1]] 
        vis1person = mne.concatenate_epochs([event1, event2],verbose=False)
        vis1person=vis1person.drop([i for i in range(numofepochs,len(vis1person))],verbose=False)
        vis1person = vis1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        vis1person = vis1person.get_data(picks=chan).flatten()
        data_Vs[counter2, counter] = vis1person
        pbar.update(1)  # Update the progress bar

        #congruent audiovisual data
        event1 = raw[AV_c_event[0]] 
        event2 = raw[AV_c_event[1]] 
        avc1person = mne.concatenate_epochs([event1, event2],verbose=False)
        avc1person=avc1person.drop([i for i in range(numofepochs,len(avc1person))],verbose=False)
        avc1person = avc1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        avc1person = avc1person.get_data(picks=chan).flatten()
        data_AVcs[counter2, counter] = avc1person
        pbar.update(1)  # Update the progress bar

        #incongruent audiovisual data
        event1 = raw[AV_ic_event[0]] 
        event2 = raw[AV_ic_event[1]] 
        avic1person = mne.concatenate_epochs([event1, event2],verbose=False)
        avic1person=avic1person.drop([i for i in range(numofepochs,len(avic1person))],verbose=False)
        avic1person = avic1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        avic1person = avic1person.get_data(picks=chan).flatten()
        data_AVics[counter2, counter] = avic1person
        pbar.update(1)  # Update the progress bar

        #counter for channels 
        counter += 1
    #counter for person
    counter2 += 1
pbar.close()  # Close the progress bar
print('-----------------------------------------')
print('Processing non-speech files...')
pbar2 = tqdm(total=14 * 36 * 4)  # Initialize the progress bar
counter2 = 0
for file in Non_speech_files:
    path = directory+"/ " + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm',verbose=False)
    raw = raw.crop(tmin=-0.1)
    counter = 0
    for chan in common:
        #auditive data
        event1 = raw[Aud_event[0]] # tagi
        event2 = raw[Aud_event[1]] # tabi
        aud1person = mne.concatenate_epochs([event1, event2],verbose=False)
        aud1person=aud1person.drop([i for i in range(numofepochs,len(aud1person))],verbose=False)
        aud1person = aud1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        aud1person = aud1person.get_data(picks=chan).flatten()
        data_A[counter2, counter] = aud1person
        pbar2.update(1)  # Update the progress bar
        #visual data
        event1 = raw[Vis_event[0]] 
        event2 = raw[Vis_event[1]]
        vis1person = mne.concatenate_epochs([event1, event2],verbose=False)
        vis1person=vis1person.drop([i for i in range(numofepochs,len(vis1person))],verbose=False)
        vis1person = vis1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        vis1person = vis1person.get_data(picks=chan).flatten()
        data_V[counter2, counter] = vis1person
        pbar2.update(1)  # Update the progress bar
        #congruent audiovisual data
        event1 = raw[AV_c_event[0]] 
        event2 = raw[AV_c_event[1]] 
        avc1person = mne.concatenate_epochs([event1, event2],verbose=False)
        avc1person=avc1person.drop([i for i in range(numofepochs,len(avc1person))],verbose=False)
        avc1person = avc1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        avc1person = avc1person.get_data(picks=chan).flatten()
        data_AVc[counter2, counter] = avc1person
        pbar2.update(1)  # Update the progress bar
        #incongruent audiovisual data
        event1 = raw[AV_ic_event[0]] 
        event2 = raw[AV_ic_event[1]] 
        avic1person = mne.concatenate_epochs([event1, event2],verbose=False)
        avic1person=avic1person.drop([i for i in range(numofepochs,len(avic1person))],verbose=False)
        avic1person = avic1person.apply_baseline(baseline=(-0.1, 0),verbose=False)
        avic1person = avic1person.get_data(picks=chan).flatten()
        data_AVic[counter2, counter] = avic1person
        pbar2.update(1)  # Update the progress bar
        #counter for channels 
        counter += 1
    #counter for person
    counter2 += 1

pbar2.close()  # Close the progress bar
"""
print(data_Vs.shape)
print(data_As.shape)
print(data_AVcs.shape)
print(data_AVics.shape)
print(data_V.shape)
print(data_A.shape)
print(data_AVc.shape)
print(data_AVic.shape)
"""








#old method
"""
# Importing data
aud1person = None
file = Speech_files[0]
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
print("Dimensions of the data for 1 person:")
print(("channels, epoch*timesteps"), np.shape(EEGdata))
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

print("")
print("Dimesions for data for all persons", np.shape(EEGdata))
"""