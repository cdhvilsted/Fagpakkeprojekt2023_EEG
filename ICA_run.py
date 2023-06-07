################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA, loadData, common, montage
from tqdm import tqdm
import time

###############################################################################

# Import data from files made in ICA_dataImport.py
data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()


# First PCA (whitening)
print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
print("")
print("# This is the first PCA: #")
print("")


reduceDimensions = 12
print("Dimensions chosen: ", 18330)
print("")
print('EEG', data_A[0].shape)
X_pca1 = np.array([])
R = np.array([])
numcomponents = []

print("doing PCA on each subject")
for i in range(0, 14):
   U, S, V, reduced_X, rho = PCA(data_A[i].T, reduceDimensions, plot=False)
   rho2 = np.diagonal(rho)
   rho2 = np.cumsum(rho2)
   if len(X_pca1) == 0:
        X_pca1 = reduced_X
        R = np.transpose(V[:,:reduceDimensions])
        R_3d = np.transpose(V[:,:reduceDimensions])
        U_3d = np.transpose(U[:,:reduceDimensions])
        numcomponents.append(np.where(rho2 > 0.95)[0][0])
        #print("Number of components for round ", i,':', numcomponents[i])
   else:
       X_pca1 = np.hstack((X_pca1, reduced_X)) 
       R = np.vstack((R, np.transpose(V[:,:reduceDimensions])))
       R_3d = np.dstack((R_3d, np.transpose(V[:,:reduceDimensions])))
       U_3d = np.dstack((U_3d, np.transpose(U[:,:reduceDimensions])))
       numcomponents.append(np.where(rho2 > 0.95)[0][0])
       #print("Number of components for round ", i,':', numcomponents[i])

print("Number of components to keep 0.95 of data after PCA1: ", max(numcomponents))


print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape, 'R:', R.shape, 'R_3d:', R_3d.shape)
X_concat = X_pca1.T
print("X_concat shape: ", X_concat.shape)

# Plotting the components and timeseries
#componentPlot(R_3d, 4, 14)
#timeSeriesPlot(U_3d, 2, 1)

print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# Second PCA
print("")
print("# This is the second PCA: #")
print("")

reduced_dim2 = 49
U, S, V, reduced_X, rho = PCA(X_concat.T, reduced_dim = reduced_dim2, plot=False)
rho2 = np.diagonal(rho)
rho2 = np.cumsum(rho2)
numcomponents = np.where(rho2 > 0.95)[0][0]
print("Number of components to keep 0.95 of data after PCA2: ", numcomponents)   


G = V #changed from U to V
print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape)
X_whithen = reduced_X
print("X shape: ", X_whithen.shape)



# Backprojecting PCA2 components into PCA1 space
Gt = np.transpose(G)
for i in range(14):
    Gt1 = Gt[:,reduceDimensions*i:reduceDimensions*(i+1)]
    #if i == 0:
    #    print('G: ', Gt1.shape)
    Rt = R_3d[:,:,i] # Basisskiftematrix ?
    comp = np.dot(Gt1, Rt) # Inverse matrix ????

    if i == 0:
        comp_3d = comp

    else:
       comp_3d = np.dstack((comp_3d, comp))

print('comp3d: ', comp_3d.shape)
#componentPlot(comp_3d, 7, 14)


print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# ICA
print("# This is the ICA step: #")
print("")

#G_ICA = V[:,:reduced_dim2]
reduced_dim3 = 46
S, A, W, sorted = ICA(X_whithen, R, G, "fastICA", reduced_dim3) #X needs shape (n_samples, n_features)

print("S shape: ", S.shape, "     A shape: ", A.shape, "     W shape: ", W.shape)

# Backprojecting ICA components into PCA1 space (first to PCA2 space)
'''
projectionPC2 = np.dot(X, np.linalg.pinv(W))
projectionPC1 = np.dot(projectionPC2, np.transpose(G)) # G is orthogonal
projection = np.dot(projectionPC1, R) # R is orthogonal'''

W_inv = np.linalg.pinv(W[sorted]) # A

Gt = np.transpose(G)
for i in range(14):
    Gt1 = Gt[:reduced_dim2,reduceDimensions*i:reduceDimensions*(i+1)]
    Rt = R_3d[:,:,i] # Basisskiftematrix ?
    
    compPC2 = np.dot(W_inv, Gt1)
    compPC1 = np.dot(compPC2, Rt)

    if i == 0:
        W_comp_3d = compPC1

    else:
       W_comp_3d = np.dstack((W_comp_3d, compPC1))

print('W_comp_3d: ', W_comp_3d.shape)
#componentPlot(W_comp_3d, 7, 14)
print(sorted)
timeSeriesPlotICA(S, sorted[0])

print("")

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")