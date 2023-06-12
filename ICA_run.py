################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage, componentTimeseriesPlot, componentTimeseriesPlotIndividual
from tqdm import tqdm
import time

###############################################################################

# Import data from files made in ICA_dataImport.py
data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()

plotTitle = 'data_A' #change according to which data is used

# First PCA (whitening)
print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
print("")
print("# This is the first PCA: #")
print("")


reduceDimensionsPCA1 = 12 # chosen from running script with and finding number of components that explain 0.95 of the variance
numberSubjects = 3 # number of subjects to run ICA on
print("Dimensions chosen: ", 18330)
print("")
print('EEG', data_A[0].shape)
X_PCA1 = np.array([]) #create empty array to store all reduced data in
Rt = np.array([]) #create empty array to store all Rt in
numcomponents_PCA1 = [] #create empty array to store number of components for each subject

print("doing PCA on each subject")
for i in range(0, numberSubjects): #looping over all 14 subjects
   U, S, V, reduced_X, rho = PCA(data_AVc[i].T, reduceDimensionsPCA1, plot=False)
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
#componentPlot(Rt_3d, 4, numberSubjects, plotTitle,list(range(0,4)))
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
"""
for i in range(numberSubjects):
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
"""
# Plotting the components
#componentPlot(PCA2_comp_3d, 4, numberSubjects, plotTitle)
sorted = list(range(168))
#componentTimeseriesPlot(PCA2_comp_3d, U, 7, numberSubjects, plotTitle, sorted)


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
#W_inv = np.linalg.pinv(W[sorted]) # A
W_inv = np.linalg.pinv(W) # A

# Backprojecting ICA components into PCA1 space (first to PCA2 space)

"""
#individual ICA
for i in range(numberSubjects):
    Gt_ind = Gt[:reduceDimensionsPCA2,reduceDimensionsPCA1*i:reduceDimensionsPCA1*(i+1)]
    Rt_ind = Rt_3d[:,:] # Basisskiftematrix ?
    
    # backprojecting ICA components into PCA2 space
    compPC2 = np.dot(W_inv, Gt_ind)
    # from PCA2 space to PCA1 space
    compPC1 = np.dot(compPC2, Rt_ind)

    # stacking the components into a 3d array
    if i == 0:
        ICA_comp_3d = compPC1
    else:
       ICA_comp_3d = np.dstack((ICA_comp_3d, compPC1))

"""
#group ICA

for i in range(numberSubjects):
    Gt_ind = Gt[:reduceDimensionsPCA2,reduceDimensionsPCA1*i:reduceDimensionsPCA1*(i+1)]
    Rt_ind = Rt_3d[:,:,i] # Basisskiftematrix ?
    
    # backprojecting ICA components into PCA2 space
    compPC2 = np.dot(W, Gt_ind)
    # from PCA2 space to PCA1 space
    compPC1 = np.dot(compPC2, Rt_ind)

    # stacking the components into a 3d array
    if i == 0:
        ICA_comp_3d = compPC1
    else:
       ICA_comp_3d = np.dstack((ICA_comp_3d, compPC1))

print('ICA_comp_3d: ', ICA_comp_3d.shape)


# Plotting the components
#componentTimeseriesPlot(ICA_comp_3d, S, 7, numberSubjects, plotTitle, sorted)
#componentTimeseriesPlotIndividual(ICA_comp_3d, S, 7, numberSubjects, plotTitle, sorted)

componentPlot(ICA_comp_3d, 7, 3, plotTitle, sorted)
#timeSeriesPlotICA(S, sorted[0], plotTitle)

print("")

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")