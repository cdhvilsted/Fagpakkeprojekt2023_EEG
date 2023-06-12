# packages
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
from picard import picard
#import fast ICA
from sklearn.decomposition import FastICA
from tqdm import tqdm
import time
#from ICA_dataImport import common, montage
import pandas as pd
#importing data
from our_group_ICA import loadData,  PCA, plotCumulativeExplainedVariances, ICA, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage, componentTimeseriesPlot, componentTimeseriesPlotIndividual


data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()

plotTitle = 'data_A' #change according to which data is used

print("")
print("# This is the first PCA: #")
print("")

reduceDimensionsPCA1 = 36 # no dimension reudction
numberSubjects = 1 # 1 subject
X_PCA1 = np.array([]) #create empty array to store all reduced data in
Rt = np.array([]) #create empty array to store all Rt in
#numcomponents_PCA1 = [] #create empty array to store number of components for each subject

#print("doing PCA on each subject")
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
        #numcomponents_PCA1.append(np.where(rho_cumsum > 0.95)[0][0])
        #print("Number of components for round ", i,':', numcomponents[i])
   else:
       X_PCA1 = np.hstack((X_PCA1, reduced_X)) 
       Rt = np.vstack((Rt, np.transpose(V[:,:reduceDimensionsPCA1]))) 
       Rt_3d = np.dstack((Rt_3d, np.transpose(V[:,:reduceDimensionsPCA1])))
       Ut_3d = np.dstack((Ut_3d, np.transpose(U[:,:reduceDimensionsPCA1])))
       #numcomponents_PCA1.append(np.where(rho_cumsum > 0.95)[0][0])
       #print("Number of components for round ", i,':', numcomponents[i])

#print("Number of components to keep 0.95 of data after PCA1: ", max(numcomponents_PCA1))
print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape, 'Rt:', Rt.shape, 'Rt_3d:', Rt_3d.shape)

componentTimeseriesPlotIndividual(Rt_3d, U, numberComponents=4, numberSubjects=1, plotTitle=plotTitle, sorted=range(0,36))

print("")
print("# This is the ICA step: #")
print("")

reduceDimensionsICA = 36 # 36*36
S, A, W, sorted = ICA(X_PCA1, "fastICA", reduced_dim=reduceDimensionsICA) #X needs shape (n_samples, n_features)
print("S shape: ", S.shape, "     A shape: ", A.shape, "     W shape: ", W.shape)

# Backprojecting ICA components into PCA1 space (first to PCA2 space)
#W_inv = np.linalg.pinv(W[sorted]) # A
W_inv = np.linalg.pinv(W) # A

for i in range(numberSubjects):
    Rt_ind = Rt_3d[:,:] # Basisskiftematrix ?
    # from PCA1 space to electrode space
    compPC1 = np.dot(Rt_ind, W_inv)

    # stacking the components into a 3d array
    if i == 0:
        ICA_comp_3d = compPC1
    else:
       ICA_comp_3d = np.dstack((ICA_comp_3d, compPC1))

print(ICA_comp_3d.shape)
# Plotting the components
componentTimeseriesPlotIndividual(ICA_comp_3d, S, 4, numberSubjects, plotTitle, sorted)

print("")

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")