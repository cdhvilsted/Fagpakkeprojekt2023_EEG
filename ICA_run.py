################# group ICA algoritm for the  EEG dataset #####################
###############################################################################

import numpy as np
from our_group_ICA import PCA, plotCumulativeExplainedVariances, ICA

###############################################################################

# Import data
from ICA_dataImport import EEGdata

reduceDimensions = 10

reduced = []
basis = []

for i in range(0, 14):
   U, S, V, reduced_X, rho = PCA(EEGdata[i], reduceDimensions, plot=False)
   print("U: ", U.shape, "S: ", S.shape, "V: ", V.shape, "reduced_X: ", reduced_X.shape, "rho: ", rho.shape)

   if len(X_pca1) == 0:

   else: X_pca1 = np.vstack((reduced_X,X_pca1))


print(X_pca1.shape)




