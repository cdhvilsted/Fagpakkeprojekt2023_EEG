import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt

# Reproducing https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html#sphx-glr-auto-examples-decomposition-plot-ica-vs-pca-py

N = 20000  # number of samples
np.random.seed(42)
rng = np.random.RandomState(42)

# Generate sine and cosine waves for the sources
time = np.linspace(0, 10, N)
S = np.column_stack([np.sin(2 * np.pi * time)*90,
                     np.cos(5* np.pi * time)*100,
                     np.sin(11 * np.pi * time)*30])

# Mix data
A1 = np.random.rand(3, 3)  # Initializing mixing matrix
X = np.dot(S, A1.T)  # Generate observations (or S * A^T)

"""
pca = PCA()
S_pca_ = pca.fit_transform(X)  # Estimate the sources
A_pca_ = pca.components_.T  # Get estimated mixing matrix

ica = FastICA(random_state=rng, whiten="arbitrary-variance", max_iter=400, tol=0.0001)
S_ica_ = ica.fit(X).transform(X)  # Estimate the sources
A_ica_ = ica.mixing_  # Get estimated mixing matrix
"""
# Comparing to simulated data

S_sim = S
X_sim = np.dot(S_sim, A1.T)  # Generate observations (or S_sim * A1)

pca_sim = PCA()
S_pca_sim = pca_sim.fit_transform(X_sim)  # Estimate the sources
A_pca_sim = pca_sim.components_.T  # Get estimated mixing matrix


ica_sim = FastICA(random_state=rng, whiten="arbitrary-variance", max_iter=400, tol=0.0001)
S_ica_sim = ica_sim.fit(X_sim).transform(X_sim)  # Estimate the sources
A_ica_sim = ica_sim.mixing_  # Get estimated mixing matrix

count = 0

for i in range(3):
    A_i = A_ica_sim[:,i]
    print("A_ica_sim[:,",i,"] :", "(",i,")   ", abs((A_i)/np.linalg.norm(A_i)))
    for j in range(3):
        A_j = A1[:,j]
        print(abs(A_j / np.linalg.norm(A_j)))
        if np.allclose(abs((A_i)/np.linalg.norm(A_i)), abs(A_j/np.linalg.norm(A_j)), atol = 0.02):
            print("True")
            count += 1
print("")
print("A_ica_sim :")
print(abs(A_ica_sim/np.linalg.norm(A_ica_sim, axis=0)))
print("")
print("A1 :")
print(abs(A1/np.linalg.norm(A1, axis=0)))
print("")
print("All components are the same" if count == 3 else "not the same")



# Plot the simulated data
plt.figure(figsize=(10, 6))
plt.plot(S_ica_sim[:, 0], label='Simulated Signal 1')
plt.plot(S_ica_sim[:, 1], label='Simulated Signal 2')
plt.plot(S_ica_sim[:, 2], label='Simulated Signal 3')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Simulated Data')
plt.legend()
plt.show()


# Plot the simulated data in normal space
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(time, X_sim[:, 0], label='Simulated Signal 1')
plt.plot(time, X_sim[:, 1], label='Simulated Signal 2')
plt.plot(time, X_sim[:, 2], label='Simulated Signal 3')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Simulated Data in Normal Space')
plt.legend()


# Plot the simulated data in ICA space
plt.subplot(4, 1, 3)
plt.plot(time,S_ica_sim[:, 0], label='Simulated Signal 1 (ICA Space)')
plt.plot(time,S_ica_sim[:, 1], label='Simulated Signal 2 (ICA Space)')
plt.plot(time,S_ica_sim[:, 2], label='Simulated Signal 3 (ICA Space)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Simulated Data in ICA Space')
plt.legend()


# Plot the simulated data in PCA space
plt.subplot(4, 1, 4)
plt.plot(time,S_pca_sim[:, 0], label='Simulated Signal 1 (PCA Space)')
plt.plot(time,S_pca_sim[:, 1], label='Simulated Signal 2 (PCA Space)')
plt.plot(time,S_pca_sim[:, 2], label='Simulated Signal 3 (PCA Space)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Simulated Data in PCA Space')
plt.legend()

# Plot the true sources
plt.subplot(4, 1, 2)
plt.plot(time, S_sim[:, 0], label='True Source 1')
plt.plot(time, S_sim[:, 1], label='True Source 2')
plt.plot(time, S_sim[:, 2], label='True Source 3')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('True Sources')
plt.legend()

plt.tight_layout()
plt.show()
