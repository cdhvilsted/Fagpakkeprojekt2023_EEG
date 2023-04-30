################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA
###############################################################################

# Import data
from ICA_dataImport import EEGdata

# First PCA
print("-------------------------------------- \033[1m ICA \033[0m --------------------------------------")
print("")
print("# This is the first PCA: #")
print("")
reduceDimensions = 10
print("Dimensions chosen: ", reduceDimensions)
print("")

X_pca1 = np.array([])
print("doing PCA on each subject")
for i in range(0, 14):
   U, S, V, reduced_X, rho = PCA(EEGdata[i], reduceDimensions, plot=False)
   if len(X_pca1) == 0:
        X_pca1 = U
   else: X_pca1 = np.concatenate((U,X_pca1))


print("U: ", U.shape, "S: ", S.shape, "V: ", V.shape, "reduced_X: ", reduced_X.shape, "rho: ", rho.shape)
R = X_pca1
print("R shape: ", R.shape)


# Second PCA (whitening)
print("")
print("# This is the second PCA: #")
print("")

U, S, V, reduced_X, rho = PCA(R.T, reduced_dim = 10, plot=False)

G = reduced_X
print("G shape: ", G.shape)


print("")
print("# This is the ICA step: #")
print("")









