# Fagpakkeprojekt 2023 - Analysis of EEG signals

This is the code for our project about Analysis of EEG signals.


= !IMPORTANT! = 

before running run the ICA_dataimport.py to get the correct datafiles



//




| Script  | Containing  |
|:----------|:----------|
| data_load.py    |  Loading the EEG signals for t-test    |
| t-testing_ERP.py    |    T-testing on grand average for Cz also Effect size is calculated  |
| ERP_plot_grand_averages.py | Plots the grand averages over ERPs |
| ICA_dataimport.py    | Imports the data, and saves it as .txt files, for faster running other scripts    | 
| ICA_individual_run.py   |  Use this for running the individulal ICA on EEG data, and get results and plots  |
| ICA_group_run.py   | Use this for running the individulal ICA on EEG data, and get results and plots   |
| our_group_ICA.py   | here we made some functions which we use a lot for ICA, and plotting   |
| t_testing_ICA.py | Results for the t-test on the synthetic channel found by ICA |
| sim_individual_ICA.py | Constructing simulated data and making individual ICA |
| sim_group_ICA.py | Constructing simulated data and making group ICA |


