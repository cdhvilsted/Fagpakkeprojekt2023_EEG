# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:15:44 2022
@author: alexa
"""
import os as os
import mne
import matplotlib.pyplot as plt
import numpy as np 


# The directory containing the preprocessed data
directory = os.path.dirname(os.path.abspath(__file__))+"/EEGproj-main/EEGproj-main/data_preproc"

#Default stuff


#A = audi
#V = vis
#tagi/tabi ord
#kongruent vs unkongruent.
Event_ids =  ["Tagi_A", "Tagi_A_Tabi_V", "Tagi_V", "Tabi_V", 
              "Tagi_A_Tagi_V", "Tabi_A", "Tabi_A_Tagi_V", "Tabi_A_Tabi_V"]

Aud_event = ["Tagi_A", "Tabi_A"] #only auditive

Vis_event = ["Tagi_V","Tabi_V"] #only visual

AV_c = ["Tagi_A_Tagi_V","Tabi_A_Tabi_V"] #congruent audivisual

AV_ic = ["Tagi_A_Tabi_V","Tabi_A_Tagi_V"] # incongruent audivisual

list_files = [i for i in os.listdir(directory) if i[-4:] == ".set"]


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

raw = mne.io.read_epochs_eeglab(directory+"/PP02_4adj.set", montage_units='dm')
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
chan = 'Cz'
#data = raw[Aud_event[0]].average().get_data(picks=chan)+raw[Aud_event[1]].average().get_data(picks=chan)
#data = data[0]
#print(data)

#data1 = raw['Tabi_A'].average().get_data(picks=chan,tmin=-1,tmax=1)
#y = data
#fig, ax = plt.subplots()
#ax.axvline(x=0.1,color='r')
#ax.plot(x,y)
#plt.show()

#for speech group
data_As=np.zeros(141,)
data_Vs=np.zeros(141,)
data_AVics=np.zeros(141,)
data_AVcs=np.zeros(141,)
#for non-speech group
data_Ans=np.zeros(141,)
data_Vns=np.zeros(141,)
data_AVicns=np.zeros(141,)
data_AVcns=np.zeros(141,)

interval = (-0.1,0)

times = np.arange(-0.1,1,step=1/128)

#getting data from speechfiles
for file in Speech_files:
    path = directory+"/ " + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw = raw.crop(tmin=-0.1)
    raw = raw.apply_baseline(interval) 
    #raw = mne.baseline.rescale(raw,times=times,baseline=interval)

    data_As += raw[Aud_event[0]].average().get_data(picks=chan)[0]+raw[Aud_event[1]].average().get_data(picks=chan)[0]
    data_Vs += raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0]
    data_AVcs +=raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0]
    data_AVics +=raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0]
    #data_V.append(raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0])
    #data_AVc.append(raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0])
    #data_AVic.append(raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0])
data_As =data_As/(len(Speech_files)*2)
data_Vs=data_Vs/(len(Speech_files)*2)
data_AVcs =data_AVcs/(len(Speech_files)*2)-data_Vs
data_AVics=data_AVics/(len(Speech_files)*2)-data_Vs
all_datas = np.concatenate((data_As,data_Vs,data_AVics,data_AVcs))

#data from non-speech
for file in Non_speech_files:
    path = directory +"/" +str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw = raw.crop(tmin=-0.1)
    print('hej')
    raw = raw.apply_baseline(interval) #one baseline method
    #raw = mne.baseline.rescale(raw,times=)
    data_Ans += raw[Aud_event[0]].average().get_data(picks=chan)[0]+raw[Aud_event[1]].average().get_data(picks=chan)[0]
    data_Vns += raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0]
    data_AVcns +=raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0]
    data_AVicns +=raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0]
    #data_V.append(raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0])
    #data_AVc.append(raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0])
    #data_AVic.append(raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0])
data_Ans =data_Ans/(len(Non_speech_files)*2)
data_Vns=data_Vns/(len(Non_speech_files)*2)
data_AVcns =data_AVcns/(len(Non_speech_files)*2)-data_Vns
data_AVicns=data_AVicns/(len(Non_speech_files)*2)-data_Vns
all_datans = np.concatenate((data_Ans,data_Vns,data_AVicns,data_AVcns))



x = np.arange(-0.1,1,step=1/128)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.axvline(x=0.1,color='r')
ax1.plot(x,data_As,color='k')
ax1.plot(x,data_AVics,color='k',linestyle='dashed')
ax1.plot(x,data_AVcs,color='0.8')
ax1.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax1.invert_yaxis()
ax2.axvline(x=0.1,color='r')
ax2.plot(x,data_Ans,color='k')
ax2.plot(x,data_AVicns,color='k',linestyle='dashed')
ax2.plot(x,data_AVcns,color='0.8')
ax2.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax2.invert_yaxis()
plt.show()



'''
fig, ax = plot.subplots()
colors = 
for i in range(len(data_A)):
    ax.plot(x,i)
ax.axvline(x=0.1,color='r')
'''
