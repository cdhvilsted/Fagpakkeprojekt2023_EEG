################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage
from tqdm import tqdm
import time
import scipy


###############################################################################

'''
# Import data from files made in ICA_dataImport.py
data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()
data_liste = [data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics]
data_titles = ['data_A', 'data_V', 'data_AVc', 'data_AVic', 'data_As', 'data_Vs', 'data_AVcs', 'data_AVics']
ICA_data_timeseries = np.empty((13677, 14, 8))
sorted_all_data = np.empty((49, 8))


for k in range(len(data_titles)):
    print('round ', k, ' of 8')
    plotTitle = data_titles[k] #change according to which data is used

    # First PCA (whitening)
    print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
    print("")
    print("# This is the first PCA: #")
    print("")


    reduceDimensionsPCA1 = 12 # chosen from running script with and finding number of components that explain 0.95 of the variance
    print("Dimensions chosen: ", 18330)
    print("")
    print('EEG', data_liste[k][0].shape)
    X_PCA1 = np.array([]) #create empty array to store all reduced data in
    Rt = np.array([]) #create empty array to store all Rt in
    numcomponents_PCA1 = [] #create empty array to store number of components for each subject

    print("doing PCA on each subject")
    for i in range(0, 14): #looping over all 14 subjects
        U, S, V, reduced_X, rho = PCA(data_liste[k][i].T, reduceDimensionsPCA1, plot=False)
        #calculate cumulative sum of explained variance to find number of components needed to explain 95% of variance
        rho_diag = np.diagonal(rho)
        rho_cumsum = np.cumsum(rho_diag) 

        #appending data to corresponding arrays
        if len(X_PCA1) == 0: #for the first subject we create the arrays
                X_PCA1 = reduced_X
                Rt = np.transpose(V[:,:reduceDimensionsPCA1]) #2d array of all Rt's
                Rt_3d = np.transpose(V[:,:reduceDimensionsPCA1]) #3d array of all Rt's with 3rd dimension being subject
                Ut_3d = np.transpose(U[:,:reduceDimensionsPCA1]) #3d array of all Ut's with 3rd dimension being subject
                numcomponents_PCA1.append(np.where(rho_cumsum > 0.95)[0][0])
                #print("Number of components for round ", i,':', numcomponents[i])
        else:
            X_PCA1 = np.hstack((X_PCA1, reduced_X)) 
            Rt = np.vstack((Rt, np.transpose(V[:,:reduceDimensionsPCA1]))) 
            Rt_3d = np.dstack((Rt_3d, np.transpose(V[:,:reduceDimensionsPCA1])))
            Ut_3d = np.dstack((Ut_3d, np.transpose(U[:,:reduceDimensionsPCA1])))
            numcomponents_PCA1.append(np.where(rho_cumsum > 0.95)[0][0])
            #print("Number of components for round ", i,':', numcomponents[i])

    print("Number of components to keep 0.95 of data after PCA1: ", max(numcomponents_PCA1))
    print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape, 'Rt:', Rt.shape, 'Rt_3d:', Rt_3d.shape)

    print("Xt_PCA1 shape: ", (X_PCA1.T).shape) # shape = (componentsPCA1*subjects, epochs*timesteps)

    # Plotting the components and timeseries
    #componentPlot(Rt_3d, 4, 14, plotTitle)
    #timeSeriesPlot(Ut_3d, 2, 1, plotTitle)

    print("")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("")


    # Second PCA
    print("")
    print("# This is the second PCA: #")
    print("")

    reduceDimensionsPCA2 = 49 # chosen from running script with and finding number of components that explain 0.95 of the variance
    U, S, V, reduced_X, rho = PCA(X_PCA1, reduced_dim = reduceDimensionsPCA2, plot=False) 
    #calculate cumulative sum of explained variance to find number of components needed to explain 95% of variance
    rho_diag = np.diagonal(rho)
    rho_cumsum = np.cumsum(rho_diag)
    numcomponents_PCA2 = np.where(rho_cumsum > 0.95)[0][0]
    print("Number of components to keep 0.95 of data after PCA2: ", numcomponents_PCA2)   


    G = V # G is equal to V from PCA2
    print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape)
    X_PCA2_whithen = reduced_X #changed name to whithen, as naming convention from ICA calls pca reduced data whithened
    print("X_PCA2_whithen shape: ", X_PCA2_whithen.shape)


    # Backprojecting PCA2 components into PCA1 space
    Gt = np.transpose(G)
    for i in range(14):
        Gt_ind = Gt[:,reduceDimensionsPCA1*i:reduceDimensionsPCA1*(i+1)] # selecting the Gt for each subject
        #if i == 0:
        #    print('G: ', Gt1.shape)
        Rt_ind = Rt_3d[:,:,i] # using Rt from PCA1 (the transposed is the inverse as it is orthogonal)
        PCA2_comp = np.dot(Gt_ind, Rt_ind) # backprojecting PCA2 components into PCA1 space

        # stacking the components into a 3d array
        if i == 0:
            PCA2_comp_3d = PCA2_comp

        else:
            PCA2_comp_3d = np.dstack((PCA2_comp_3d, PCA2_comp))

    print('PCA2_comp_3d: ', PCA2_comp_3d.shape)
    # Plotting the components
    #componentPlot(PCA2_comp_3d, 7, 14, plotTitle)


    print("")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("")


    # ICA
    print("# This is the ICA step: #")
    print("")

    # should the reduceDimensionsICA be used? is not used in the function
    reduceDimensionsICA = 46
    S, A, W, sorted = ICA(X_PCA2_whithen, Rt, G, "fastICA", reduced_dim=reduceDimensionsICA) #X needs shape (n_samples, n_features)

    print("S shape: ", S.shape, "     A shape: ", A.shape, "     W shape: ", W.shape)

    # Backprojecting ICA components into PCA1 space (first to PCA2 space)
    W_inv = np.linalg.pinv(W) # A

    # Backprojecting ICA components into PCA1 space (first to PCA2 space)
    for i in range(14):
        Gt_ind = Gt[:reduceDimensionsPCA2,reduceDimensionsPCA1*i:reduceDimensionsPCA1*(i+1)]
        Rt_ind = Rt_3d[:,:,i] # Basisskiftematrix ?
        
        # backprojecting ICA components into PCA2 space
        compPC2 = np.dot(W_inv, Gt_ind)
        # from PCA2 space to PCA1 space
        compPC1 = np.dot(compPC2, Rt_ind)

        # stacking the components into a 3d array
        if i == 0:
            ICA_comp_3d = compPC1
        else:
            ICA_comp_3d = np.dstack((ICA_comp_3d, compPC1))

    print('ICA_comp_3d: ', ICA_comp_3d.shape)

    # Plotting the components
    #componentPlot(ICA_comp_3d, 7, 14, plotTitle)
    #timeSeriesPlotICA(S, sorted[0], plotTitle)
    for i in range(14):
        ICA_comp_3d_ind = ICA_comp_3d[:,:,i]
        ICA_data_timeseries[:,i,k] = np.dot(ICA_comp_3d_ind[sorted[0],:], data_liste[k][i])
    
    
    print("")

    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

print('done')

'''

#t-testing begins

from ERP_plot_grand_averages import S_A, S_V, S_AVc, S_AVic, S_As, S_Vs, S_AVcs, S_AVics

print('Shape of S_As: ', S_As.shape)

ICA_data_timeseries = np.concatenate((S_A, S_V, S_AVc, S_AVic, S_As, S_Vs, S_AVcs, S_AVics), axis=2)

print('Shape of ICA_data_timeseries: ', ICA_data_timeseries.shape)

times = np.arange(-0.1,1,step=1/128)
N_comp_min = (np.where(times <= 0.05))[-1][-1] # 50 ms

N_comp_max = (np.where(times >= 0.15))[0][0] # 150 ms
#print('N indexes: ' + str(N_comp_min) + ',' + str(N_comp_max))

# P2 component
P_comp_min = (np.where(times <= 0.15))[-1][-1] # 150 ms
P_comp_max = (np.where(times >= 0.25))[0][0] # 250 ms
#print('P indexes: ' + str(P_comp_min) + ',' + str(P_comp_max))

# amplitude samples for same stimuli
N1_As = []
N1_ics = []
N1_cs = []
N1_Vs = []

P2_As = []
P2_ics = []
P2_cs = []
P2_Vs = []

N1_Ans = []
N1_icns = []
N1_cns = []
N1_Vns = []

P2_Ans = []
P2_icns = []
P2_cns = []
P2_Vns = []

for i in range(14):
    # taking average over epochs
    t_data_A = np.mean(ICA_data_timeseries[i,:,0].reshape(97,141), axis = 0) # 97 epochs, 141 timesteps
    t_data_V = np.mean(ICA_data_timeseries[i,:,1].reshape(97,141), axis = 0)
    t_data_AVc = np.mean(ICA_data_timeseries[i,:,2].reshape(97,141), axis = 0)
    t_data_AVic = np.mean(ICA_data_timeseries[i,:,3].reshape(97,141), axis = 0)
    t_data_As = np.mean(ICA_data_timeseries[i,:,4].reshape(97,141), axis = 0)
    t_data_Vs = np.mean(ICA_data_timeseries[i,:,5].reshape(97,141), axis = 0)
    t_data_AVcs = np.mean(ICA_data_timeseries[i,:,6].reshape(97,141), axis = 0)
    t_data_AVics = np.mean(ICA_data_timeseries[i,:,7].reshape(97,141), axis = 0)
    
    #non-speech 
    N1_Ans.append(np.min(t_data_A[N_comp_min:N_comp_max]))
    N1_Vns.append(np.min(t_data_V[N_comp_min:N_comp_max]))
    N1_cns.append(np.min(t_data_AVc[N_comp_min:N_comp_max]))
    N1_icns.append(np.min(t_data_AVic[N_comp_min:N_comp_max]))
    P2_Ans.append(np.max(t_data_A[P_comp_min:P_comp_max]))
    P2_Vns.append(np.max(t_data_V[P_comp_min:P_comp_max]))
    P2_cns.append(np.max(t_data_AVc[P_comp_min:P_comp_max]))
    P2_icns.append(np.max(t_data_AVic[P_comp_min:P_comp_max]))

    #speech
    N1_As.append(np.min(t_data_As[N_comp_min:N_comp_max]))
    N1_Vs.append(np.min(t_data_Vs[N_comp_min:N_comp_max]))
    N1_cs.append(np.min(t_data_AVcs[N_comp_min:N_comp_max]))
    N1_ics.append(np.min(t_data_AVics[N_comp_min:N_comp_max]))
    P2_As.append(np.max(t_data_As[P_comp_min:P_comp_max]))
    P2_Vs.append(np.max(t_data_Vs[P_comp_min:P_comp_max]))
    P2_cs.append(np.max(t_data_AVcs[P_comp_min:P_comp_max]))
    P2_ics.append(np.max(t_data_AVics[P_comp_min:P_comp_max]))



# N1, speech
N1_ics_nv = [N1_ics[i]-N1_Vs[i] for i in range(len(N1_ics))]
N1_cs_nv = [N1_cs[i]-N1_Vs[i] for i in range(len(N1_cs))]
test_As_in = scipy.stats.ttest_rel(N1_As, N1_ics_nv) # auditiv + incongruent
test_As_con = scipy.stats.ttest_rel(N1_As, N1_cs_nv) # auditiv + congruent

# P2, speech
P2_ics_nv = [P2_ics[i]-P2_Vs[i] for i in range(len(P2_ics))]
P2_cs_nv = [P2_cs[i]-P2_Vs[i] for i in range(len(P2_cs))]
As_in = scipy.stats.ttest_rel(P2_As, P2_ics_nv) # auditiv + incongruent
As_con = scipy.stats.ttest_rel(P2_As, P2_cs_nv) # auditiv + congruent

# N1, non-speech
N1_icns_nv = [N1_icns[i]-N1_Vns[i] for i in range(len(N1_icns))]
N1_cns_nv = [N1_cns[i]-N1_Vns[i] for i in range(len(N1_cns))]
test_Ans_in = scipy.stats.ttest_rel(N1_Ans, N1_icns_nv ) # auditiv + incongruent
test_Ans_con = scipy.stats.ttest_rel(N1_Ans, N1_cns_nv) # auditiv + congruent

# P2, non-speech
P2_icns_nv = [P2_icns[i]-P2_Vns[i] for i in range(len(P2_icns))]
P2_cns_nv = [P2_cns[i]-P2_Vns[i] for i in range(len(P2_cns))]
Ans_in = scipy.stats.ttest_rel(P2_Ans, P2_icns_nv) # auditiv + incongruent
Ans_con = scipy.stats.ttest_rel(P2_Ans, P2_cns_nv) # auditiv + congruent

print('-----------------------------------------')
print('------------ t-test p-values ------------')
print(' ')
print('---------------- SPEECH -----------------')
print('------------------ N1 -------------------')
print('T-test for A and incongruent AV in speech mode gives (N): ', test_As_in.pvalue)
print('T-test for A and congruent AV in speech mode gives (N): ', test_As_con.pvalue)
print('------------------ P2 -------------------')
print('T-test for A and incongruent AV in speech mode gives (P): ', As_in.pvalue)
print('T-test for A and congruent AV in speech mode gives (P): ', As_con.pvalue)
print(' ')
print('-------------- NON-SPEECH ---------------')
print('------------------ N1 -------------------')
print('T-test for A and incongruent AV in non-speech mode gives (N): ', test_Ans_in.pvalue)
print('T-test for A and congruent AV in non-speech mode gives (N): ', test_Ans_con.pvalue)
print('------------------ P2 -------------------')
print('T-test for A and incongruent AV in non-speech mode gives (P): ', Ans_in.pvalue)
print('T-test for A and congruent AV in non-speech mode gives (P): ', Ans_con.pvalue)
