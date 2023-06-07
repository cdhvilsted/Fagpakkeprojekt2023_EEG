import os as os
import mne
import matplotlib.pyplot as plt
import numpy as np 


# The directory containing the preprocessed data
#direct = "/Users/ceciliehvilsted/Documents/GitHub/Fagpakkeprojekt2023_EEG/EEGproj-main/EEGproj-main/data_preproc"
directory = os.path.dirname(os.path.abspath(__file__))+"/EEGproj-main/EEGproj-main/data_preproc"

#print("Working directory:  ", directory[:30],"/.../",directory[-57:], " ") # just printing the path to the data

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
print("list_files contains a list of all the data files: ",list_files[:2], " ... ", list_files[-2:])

# The list of participants
Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", 
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]
Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19", 
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]

# The list of channels
common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']
# The list of files in
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
#raw = mne.io.read_epochs_eeglab("/Users/ceciliehvilsted/Documents/GitHub/Fagpakkeprojekt2023_EEG/EEGproj-main/EEGproj-main/data_preproc/PP02_4adj.set", montage_units='dm')

gennemsnit = raw.average(method='mean',by_event_type=True)
chan = 'Cz'

#for speech group
data_As=np.zeros(141,)
data_Vs=np.zeros(141,)
data_AVics=np.zeros(141,)
data_AVcs=np.zeros(141,)

data_As_ind = []
data_Vs_ind =[]
data_AVics_ind =[]
data_AVcs_ind=[]

#for non-speech group
data_Ans=np.zeros(141,)
data_Vns=np.zeros(141,)
data_AVicns=np.zeros(141,)
data_AVcns=np.zeros(141,)

data_Ans_ind=[]
data_Vns_ind=[]
data_AVicns_ind=[]
data_AVcns_ind=[]


#for speech group
data_As=np.zeros((36,14,141))
data_Vs=np.zeros((36,14,141))
data_AVics=np.zeros((36,14,141))
data_AVcs=np.zeros((36,14,141))

data_As_ind = []
data_Vs_ind =[]
data_AVics_ind =[]
data_AVcs_ind=[]

#for non-speech group
data_Ans=np.zeros((36,141))
data_Vns=np.zeros((36,141))
data_AVicns=np.zeros((36,141))
data_AVcns=np.zeros((36,141))

data_Ans_ind=[]
data_Vns_ind=[]
data_AVicns_ind=[]
data_AVcns_ind=[]


interval = (-0.1, 0)

times = np.arange(-0.1,1,step=1/128)
baseline = interval
#raw.average().plot()

#getting data from speechfiles
# raw = epochs
#data from one person auditive
'''
path = directory + '/' +str(Speech_files[0])
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path,montage_units='dm')
raw = raw.crop(tmin=-0.1) 
#print(len(raw))
data1 = np.zeros((len(raw),len(common),141))


for i in range(len(raw)):
    for k in range(len(common)):
        data1[i][k]=(raw[Aud_event[0]][i].apply_baseline(baseline).get_data(picks=common[k])[0]+raw[Aud_event[1]][i].apply_baseline(baseline).get_data(picks=common[k])[0])/2
'''

for file in Speech_files:
    path = directory +"/"+ str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw = raw.crop(tmin=-0.1) 
    #raw = raw.apply_baseline(interval) 
    #raw = mne.baseline.rescale(raw,times=times,baseline=(-0.1,0), mode='mean', copy=False)
    #raw.set_channel_types({'event': 'stim'})
    #raw.pick_types(meg=False, stim=False, eog=False, eeg=True)
    #events = mne.events_from_annotations(raw, event_id=Event_ids)
    #epochs = mne.Epochs(raw, events, event_id=Event_ids, baseline=None, preload=True)
    #raw.apply_baseline(interval) 
    #epochs.apply_function(fun=mne.baseline.rescale, times=times, baseline=(-0.1, 0), mode='mean')
    #data = epochs.get_data()
    #raw_array = mne.baseline.rescale(raw,times=times, baseline = (-0.1,0), mode='mean', copy=False)  
    
    #raw_array = mne.time_frequency.tfr_array_morlet(raw.get_data(), sfreq=128,freqs=freqs, n_cycles=freqs/2,output='avg_power')
    # Baseline the output
    #mne.baseline.rescale(raw_array, raw.times, (-0.1, 1), mode='mean', copy=False)

    data_As += raw[Aud_event[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[Aud_event[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_Vs += raw[Vis_event[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[Vis_event[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_AVcs +=raw[AV_c[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_c[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_AVics +=raw[AV_ic[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_ic[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    #data_V.append(raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0])
    #data_AVc.append(raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0])
    #data_AVic.append(raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0])
    
    data_As_ind.append((raw[Aud_event[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[Aud_event[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    data_Vs_ind.append((raw[Vis_event[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[Vis_event[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    data_AVcs_ind.append((raw[AV_c[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_c[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    data_AVics_ind.append((raw[AV_ic[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_ic[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    
'''
data_As =data_As/len(Speech_files)
data_Vs=data_Vs/len(Speech_files)
data_AVcs =data_AVcs/len(Speech_files)-data_Vs
data_AVics=data_AVics/len(Speech_files)-data_Vs
'''

data_As =data_As/(len(Speech_files)*2)
data_Vs=data_Vs/(len(Speech_files)*2)
data_AVcs =data_AVcs/(len(Speech_files)*2)-data_Vs
data_AVics=data_AVics/(len(Speech_files)*2)-data_Vs

all_datas = np.concatenate((data_As,data_Vs,data_AVics,data_AVcs))
#mne.baseline.rescale(all_datas.get_data(),times=times,baseline=(-0.1,0), mode='mean', copy=False)

#data from non-speech
for file in Non_speech_files:
    path = directory +"/" + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw = raw.crop(tmin=-0.1)
    #raw.apply_baseline(interval) #one baseline method
    #raw_array = mne.baseline.rescale(raw,times=times, baseline = (-0.1,0), mode='mean', copy=False)
    data_Ans += raw[Aud_event[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[Aud_event[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_Vns += raw[Vis_event[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[Vis_event[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_AVcns +=raw[AV_c[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_c[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_AVicns +=raw[AV_ic[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_ic[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    #data_V.append(raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0])
    #data_AVc.append(raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0])
    #data_AVic.append(raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0])
    
    data_Ans_ind.append((raw[Aud_event[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[Aud_event[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    data_Vns_ind.append((raw[Vis_event[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[Vis_event[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    data_AVcns_ind.append((raw[AV_c[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_c[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    data_AVicns_ind.append((raw[AV_ic[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_ic[1]].apply_baseline(baseline).get_data(picks=chan)[0])/2)
    

data_Ans =data_Ans/(len(Non_speech_files)*2)
data_Vns=data_Vns/(len(Non_speech_files)*2)
data_AVcns =data_AVcns/(len(Non_speech_files)*2)-data_Vns
data_AVicns=data_AVicns/(len(Non_speech_files)*2)-data_Vns
all_datans = np.concatenate((data_Ans,data_Vns,data_AVicns,data_AVcns))


'''
#trial for all people

k = 0
for file in Speech_files:
    path = directory +"/"+ str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw = raw.crop(tmin=-0.1) 
    for i in range(len(common)):
        data_As[i][k] += (raw[Aud_event[0]].average().apply_baseline(baseline).get_data(picks=common[i])[0]+raw[Aud_event[1]].average().apply_baseline(baseline).get_data(picks=common[i])[0])/2
        data_Vs[i][k] += (raw[Vis_event[0]].average().apply_baseline(baseline).get_data(picks=common[i])[0]+raw[Vis_event[1]].average().apply_baseline(baseline).get_data(picks=common[i])[0])/2
        data_AVcs[i][k] +=(raw[AV_c[0]].average().apply_baseline(baseline).get_data(picks=common[i])[0]+raw[AV_c[1]].average().apply_baseline(baseline).get_data(picks=common[i])[0])/2
        data_AVics[i][k] +=(raw[AV_ic[0]].average().apply_baseline(baseline).get_data(picks=common[i])[0]+raw[AV_ic[1]].average().apply_baseline(baseline).get_data(picks=common[i])[0])/2
    k +=1

data_AVcs =data_AVcs-data_Vs
data_AVics=data_AVics-data_Vs




#data_As =data_As/(len(Speech_files)*2)
#data_Vs=data_Vs/(len(Speech_files)*2)
#data_AVcs =data_AVcs/(len(Speech_files)*2)-data_Vs
#data_AVics=data_AVics/(len(Speech_files)*2)-data_Vs

#all_datas = np.dstack((data_As,data_Vs,data_AVics,data_AVcs))
#print(np.shape(all_datas))
#mne.baseline.rescale(all_datas.get_data(),times=times,baseline=(-0.1,0), mode='mean', copy=False)

#data from non-speech
for file in Non_speech_files:
    path = directory +"/" + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw = raw.crop(tmin=-0.1)
    #raw.apply_baseline(interval) #one baseline method
    #raw_array = mne.baseline.rescale(raw,times=times, baseline = (-0.1,0), mode='mean', copy=False)
    data_Ans += raw[Aud_event[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[Aud_event[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_Vns += raw[Vis_event[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[Vis_event[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_AVcns +=raw[AV_c[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_c[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    data_AVicns +=raw[AV_ic[0]].average().apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_ic[1]].average().apply_baseline(baseline).get_data(picks=chan)[0]
    #data_V.append(raw[Vis_event[0]].average().get_data(picks=chan)[0]+raw[Vis_event[1]].average().get_data(picks=chan)[0])
    #data_AVc.append(raw[AV_c[0]].average().get_data(picks=chan)[0]+raw[AV_c[1]].average().get_data(picks=chan)[0])
    #data_AVic.append(raw[AV_ic[0]].average().get_data(picks=chan)[0]+raw[AV_ic[1]].average().get_data(picks=chan)[0])
    
    data_Ans_ind.append(raw[Aud_event[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[Aud_event[1]].apply_baseline(baseline).get_data(picks=chan)[0])
    data_Vns_ind.append(raw[Vis_event[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[Vis_event[1]].apply_baseline(baseline).get_data(picks=chan)[0])
    data_AVcns_ind.append(raw[AV_c[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_c[1]].apply_baseline(baseline).get_data(picks=chan)[0])
    data_AVicns_ind.append(raw[AV_ic[0]].apply_baseline(baseline).get_data(picks=chan)[0]+raw[AV_ic[1]].apply_baseline(baseline).get_data(picks=chan)[0])
    

data_Ans =data_Ans/(len(Non_speech_files)*2)
data_Vns=data_Vns/(len(Non_speech_files)*2)
data_AVcns =data_AVcns/(len(Non_speech_files)*2)-data_Vns
data_AVicns=data_AVicns/(len(Non_speech_files)*2)-data_Vns
all_datans = np.concatenate((data_Ans,data_Vns,data_AVicns,data_AVcns))
#mne.baseline.rescale(all_datans.get_data(),times=times,baseline=(-0.1,0), mode='mean', copy=False)
'''
