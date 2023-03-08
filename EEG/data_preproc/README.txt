
 SUBJECT INFO
 Speech mode: PP03, PP09, PP10, PP11, PP12, PP13, PP14, PP15, PP16, PP17, PP20, PP25, PP26, PP28
 Non-speech mode: PP02, PP04, PP05, PP06, PP07, PP08, PP18, PP19, PP21, PP22, PP23, PP24, PP27, PP29


 PREPROCESSING PIPELINE

 - highpass filtering at 0.5 Hz
 - downsampling to 128 Hz
 - reject bad channels by kurtosis
 - lowpass filter at 40 Hz
 - time-lock to sound onset
 - thresholding on non-frontal channels - epochs with |amplitude| > 150 muV rejected
 - run ICA
 - run ICMARC
 - save an EEGlab subject_group_4adj.set file (in data_preproc)
 - semi-automatic IC rejection aided by ICMARC (saved in data_icmarked)