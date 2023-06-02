% Load EEGLAB toolbox
import eeglab;
addpath('/Users/idaraagart/Library/CloudStorage/OneDrive-Personligt/DTU/4th_semester/fagprojekt/Fagpakkeprojekt2023_EEG/EEGproj-main/EEGproj-main/data_preproc')



% Set directory where EEG data files are stored
data_dir = './';

% Load each EEG data file and store in a cell array
EEG = cell(length(Speech_files), 1);
for i = 1:length(Speech_files)
    filename = fullfile(data_dir, Speech_files{i});
    EEG{i} = pop_loadset(filename);
end

