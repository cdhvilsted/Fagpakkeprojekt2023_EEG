################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA, pvaf, componentPlot,timeSeriesPlot, timeSeriesPlotICA
from tqdm import tqdm
import time

###############################################################################

# Import data
from ICA_dataImport import data_A, montage, common

# First PCA (whitening)
print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
print("")
print("# This is the first PCA: #")
print("")


reduceDimensions = 10
print("Dimensions chosen: ", 18330)
print("")
print('EEG', data_A[0].shape)
X_pca1 = np.array([])
R = np.array([])
print("doing PCA on each subject")
for i in range(0, 14):
   U, S, V, reduced_X, rho = PCA(data_A[i].T, reduceDimensions, plot=False)
   if len(X_pca1) == 0:
        X_pca1 = reduced_X
        R = np.transpose(V[:,:reduceDimensions])
        R_3d = np.transpose(V[:,:reduceDimensions])
        U_3d = np.transpose(U[:,:reduceDimensions])

   else:
       X_pca1 = np.hstack((X_pca1, reduced_X)) 
       R = np.vstack((R, np.transpose(V[:,:reduceDimensions])))
       R_3d = np.dstack((R_3d, np.transpose(V[:,:reduceDimensions])))
       U_3d = np.dstack((U_3d, np.transpose(U[:,:reduceDimensions])))



print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape, 'R:', R.shape, 'R_3d:', R_3d.shape)
X_concat = X_pca1.T
print("X_concat shape: ", X_concat.shape)

# Plotting the components and timeseries
componentPlot(R_3d, 4, 14)
#timeSeriesPlot(U_3d, 2, 1)

print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# Second PCA
print("")
print("# This is the second PCA: #")
print("")

reduced_dim2 = 140
U, S, V, reduced_X, rho = PCA(X_concat.T, reduced_dim = reduced_dim2, plot=False)

G = V #changed from U to V
print("U: ", U.shape, "     S: ", S.shape, "     V: ", V.shape, "\nreduced_X: ", reduced_X.shape, "     rho: ", rho.shape)
X_whithen = reduced_X
print("X shape: ", X_whithen.shape)

# Backprojecting PCA2 components into PCA1 space
Gt = np.transpose(G)
for i in range(14):
    Gt1 = Gt[:,reduceDimensions*i:reduceDimensions*(i+1)]
    Rt = R_3d[:,:,i] # Basisskiftematrix ?
    comp = np.dot(Gt1, Rt) # Inverse matrix ????
    
    if i == 0:
        comp_3d = comp

    else:
       comp_3d = np.dstack((comp_3d, comp))

print('comp3d: ', comp_3d.shape)
#componentPlot(comp_3d, 4, 14)


print("")
print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("")


# ICA
print("# This is the ICA step: #")
print("")

S, A, W, sorted = ICA(X_whithen, R, G, "fastICA") #X needs shape (n_samples, n_features)


print("S shape: ", S.shape, "     A shape: ", A.shape, "     W shape: ", W.shape)

print("")

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")


#try to plot ESP from S
S = S.T
#timeSeriesPlotICA(S, sorted[-1])

back_Y = np.zeros((14,140,36))

print(np.transpose(R_3d[:,:,0]).shape, 'should be 36,10')
#print(np.transpose(G[:,10*(0):10*(0+1)]).shape, 'should be 10, 140')
print(G[10*(0):10*(0+1),:].shape, 'should be 10, 140')

print(np.linalg.pinv(W).shape, 'should be 140, 140')
print((np.transpose(R_3d[:,:,0]) @ np.transpose(G[:,10*(0):10*(0+1)]) @ np.linalg.pinv(W)).shape)
for i in range(14):
    back_Y[i,:,:] = np.transpose(np.transpose(R_3d[:,:,i]) @ G[10*(0):10*(0+1),:] @ np.linalg.pinv(W))

'''
for i in range(14):
    for j in range(10):
        #back_Y[i, j, :] = np.matmul(A @ S[:, i].reshape((-1, 1)), np.ones((1, 36)))
        back_Y[i, j, :] = np.transpose(R_3d[:,:,j]) @ np.transpose(G[:,10*(i):10*(i+1)]) @ np.linalg.pinv(W)
'''
print(np.shape(back_Y))

# data for hver forsøgsperson kommer af at gange mixing matrix med source for hver forsøgsperson fx X0 = A[0,0,:] @ S[0,:]

# sorter efter varians i hver component
#print(A[0,0,:])
#var_S = np.var(S, axis=1)
#print(np.argsort(var_S)) # sort index from smallest to highest variance