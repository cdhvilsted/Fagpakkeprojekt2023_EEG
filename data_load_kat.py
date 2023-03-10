# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:15:44 2022
@author: alexa
"""
import os as os
import mne
import matplotlib.pyplot as plt


# The directory containing the preprocessed data
direct = "EEGproj-main\EEGproj-main\pipeline_data\data_preproc"

#Default stuff


#A = audi
#V = vis
#tagi/tabi ord
#kongruent vs unkongruent.
Event_ids =  ["Tagi_A", "Tagi_A_Tabi_V", "Tagi_V", "Tabi_V", 
              "Tagi_A_Tagi_V", "Tabi_A", "Tabi_A_Tagi_V", "Tabi_A_Tabi_V"]



list_files = [i for i in os.listdir(direct) if i[-4:] == ".set"]


Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", 
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]
Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19", 
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]

common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']

Speech_files = [ i + "_4adj.set" for i in Speech]
Non_speech_files = [i + "_4adj.set" for i in Non_speech]


#skal ikke bruges endnu
event_dict = { 'visual/b/high' : 1, 
               'visual/b/mid'  : 2, 
               'visual/b/low'  : 3,
               'visual/g/high' : 4, 
               'visual/g/mid'  : 5, 
               'visual/g/low'  : 6,
               'auditory/b/high' : 7,
               'auditory/b/mid'  : 8, 
               'auditory/b/low'  : 9,
               'auditory/g/high' : 10,
               'auditory/g/mid'  : 11, 
               'auditory/g/low'  : 12,
               'audiovisual/b/high/congruent': 13, 
               'audiovisual/b/med/congruent' : 14,
               'audiovisual/b/low/congruent' : 15,
               'audiovisual/g/high/congruent': 16, 
               'audiovisual/g/med/congruent' : 17,
               'audiovisual/g/low/congruent' : 18,
               'audiovisual/bg/high' : 19,
               'audiovisual/bg/med'  : 20,
               'audiovisual/bg/low'  : 21,
               'NA1'                  : 22,   # Remove !!
               'NA2'                  : 23,   # Remove !!
               'NA3'                  : 24  } # Remove !!

raw = mne.io.read_epochs_eeglab("EEGproj-main\EEGproj-main\pipeline_data\data_preproc\PP02_4adj.set", montage_units='dm')
#raw = mne.io.Raw("EEGproj-main\EEGproj-main\pipeline_data\data_preproc\PP02_4adj.set")
#fig = plt.figure(raw.plot())
#plt.show()
#fig = raw.plot_sensors(show_names=True)
#fig = raw.plot_sensors('3d')
#fig2 = plt.figure(raw.plot_sensors(show_names=True,sphere='eeglab'))
#plt.show()
#raw.copy().pick_types(meg=False,stim=True).plot(start=3,duration=6)
#events = mne.find_events(raw)
#epochs = mne.Epochs(raw,Event_ids, event_id=event_dict)
#fig3 = raw.plot_projs_topomap()
#plt.show()
#epochs.plot_projs_topomap()
#for pro in (False, True):
 #   with mne.viz.use_browser_backend('matplotlib'):
  #      fig = raw['Tagi_A'].plot(n_channels=5,scalings=dict(eeg=50e-6),show_scrollbars=False)
   # fig.subplots_adjust(top=0.9)
    #ref = 'Average' if pro else 'No'
    #fig.suptitle(f'{ref} reference',size='xx-large',weight='bold')
#plt.show()
gennemsnit = raw.average(method='mean',by_event_type=True)
#print(len(gennemsnit))
#fig = plt.figure(gennemsnit[1].plot())
#fig2 = plt.figure(gennemsnit[2].plot())
#plt.show()
#fig1 = raw['Tagi_A'].average().plot_topomap(times=[0.1],sphere='eeglab')
#plt.show()
data = raw['Tagi_A'].average().get_data(picks='Cz')
data1 = raw['Tabi_A'].average().get_data(picks='Cz')
print(data[0][130])
