
import numpy as np
from sklearn.decomposition import PCA, FastICA

# Reproducing https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html#sphx-glr-auto-examples-decomposition-plot-ica-vs-pca-py

N = 20000 # antal samples

rng = np.random.RandomState(42)
S = rng.standard_t(1.5, size=(N, 36))
S[:, 0] *= 2.0

# Mix data
A1 = np.random.rand(36, 36) # Initiallizing mixing matrix
X = np.dot(S, A1.T)  # Generate observations (or S * A^T)

ica = FastICA(random_state=rng, whiten="arbitrary-variance")
S_ica_ = ica.fit(X).transform(X)  # Estimate the sources
#S_ica_ /= S_ica_.std(axis=0)
A_ica_ = ica.mixing_  # Get estimated mixing matrix

# Comparing to simulated data

S_sim = np.random.normal(size=(N, 36))
S_sim[:, -1] *= 2.0

X_sim = np.dot(S_sim, A1.T)

ica_sim = FastICA(random_state=rng, whiten="arbitrary-variance", max_iter=200, tol=0.0001)
S_ica_sim = ica_sim.fit(X_sim).transform(X_sim)  # Estimate the sources
S_ica_sim /= S_ica_sim.std(axis=0)
A_ica_sim = ica_sim.mixing_  # Get estimated mixing matrix

count = 0
for i in range(36):
    A_i = A_ica_[i]
    for j in range(36):
        A_j = A_ica_sim[j]
        if np.allclose(abs(A_i/np.linalg.norm(A_i)), abs(A_j/np.linalg.norm(A_j))):
            count += 1

print(count)