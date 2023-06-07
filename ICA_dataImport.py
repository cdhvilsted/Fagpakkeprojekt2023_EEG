# for updating the files of the matrix

import mne
import numpy as np
import random
import os
from tqdm import tqdm
from numpy import savetxt

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

saves = ["data_A", "data_V", "data_AVc", "data_AVic", "data_As", "data_Vs", "data_AVcs", "data_AVics"]

for save in saves:
    reshaped_data = globals()[save].reshape(globals()[save].shape[0], -1)
    filename = save + ".txt"
    savetxt(filename, reshaped_data, delimiter=',')
    print("Saved " + filename)





"""
saves = ["data_A","data_V","data_AVc","data_AVic","data_As","data_Vs","data_AVcs","data_AVics"]

data_A_reshaped = data_A.reshape(data_A.shape[0], -1)
# save data_A as a .txt
savetxt("data_A.txt", data_A_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_A = np.loadtxt("data_A.txt",delimiter=',')

load_3d_data_A = loaded_2d_data_A.reshape(
    loaded_2d_data_A.shape[0], loaded_2d_data_A.shape[1] // data_A.shape[2], data_A.shape[2])



data_V_reshaped = data_V.reshape(data_V.shape[0], -1)
# save data_V as a .txt
savetxt("data_V.txt", data_V_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_V = np.loadtxt("data_V.txt",delimiter=',')

load_3d_data_V = loaded_2d_data_V.reshape(
    loaded_2d_data_V.shape[0], loaded_2d_data_V.shape[1] // data_V.shape[2], data_V.shape[2])


data_AVc_reshaped = data_AVc.reshape(data_AVc.shape[0], -1)
# save data_V as a .txt
savetxt("data_AVc.txt", data_AVc_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_AVc = np.loadtxt("data_AVc.txt",delimiter=',')

load_3d_data_AVc = loaded_2d_data_AVc.reshape(
    loaded_2d_data_AVc.shape[0], loaded_2d_data_AVc.shape[1] // data_AVc.shape[2], data_AVc.shape[2])



data_AVic_reshaped = data_AVic.reshape(data_AVic.shape[0], -1)
# save data_Vic as a .txt
savetxt("data_AVic.txt", data_AVic_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_AVic = np.loadtxt("data_AVic.txt",delimiter=',')

load_3d_data_AVic = loaded_2d_data_AVic.reshape(
    loaded_2d_data_AVic.shape[0], loaded_2d_data_AVic.shape[1] // data_AVic.shape[2], data_AVic.shape[2])


data_As_reshaped = data_As.reshape(data_As.shape[0], -1)
# save data_A as a .txt
savetxt("data_As.txt", data_As_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_As = np.loadtxt("data_As.txt",delimiter=',')

load_3d_data_As = loaded_2d_data_As.reshape(
    loaded_2d_data_As.shape[0], loaded_2d_data_As.shape[1] // data_As.shape[2], data_As.shape[2])



data_Vs_reshaped = data_Vs.reshape(data_Vs.shape[0], -1)
# save data_Vs as a .txt
savetxt("data_Vs.txt", data_Vs_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_Vs = np.loadtxt("data_Vs.txt",delimiter=',')

load_3d_data_Vs = loaded_2d_data_Vs.reshape(
    loaded_2d_data_Vs.shape[0], loaded_2d_data_Vs.shape[1] // data_Vs.shape[2], data_Vs.shape[2])


data_AVcs_reshaped = data_AVcs.reshape(data_AVcs.shape[0], -1)
# save data_Vcs as a .txt
savetxt("data_AVcs.txt", data_AVcs_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_AVcs = np.loadtxt("data_AVcs.txt",delimiter=',')

load_3d_data_AVcs = loaded_2d_data_AVcs.reshape(
    loaded_2d_data_AVcs.shape[0], loaded_2d_data_AVcs.shape[1] // data_AVcs.shape[2], data_AVcs.shape[2])



data_AVics_reshaped = data_AVics.reshape(data_AVics.shape[0], -1)
# save data_Vics as a .txt
savetxt("data_AVics.txt", data_AVics_reshaped, delimiter=',')

# retrieving data from file.
loaded_2d_data_AVics = np.loadtxt("data_AVics.txt",delimiter=',')

load_3d_data_AVics = loaded_2d_data_AVics.reshape(
    loaded_2d_data_AVics.shape[0], loaded_2d_data_AVics.shape[1] // data_AVics.shape[2], data_AVics.shape[2])
"""
