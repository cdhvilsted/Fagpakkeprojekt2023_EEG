import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import random

# Set the random seed to 42
np.random.seed(42)

# Generate simulated signals
num_subjects = 2  # Number of subjects
num_samples = 200  # Number of samples per subject
num_sources = 3  # Number of independent sources per subject
num_mixtures = num_sources  # Number of observed mixtures per subject

# Create random mixing matrices for each subject
mixing_matrices = [np.random.rand(num_mixtures, num_sources) for _ in range(num_subjects)]


# Generate random independent sources for each subject
sources = []
for i in range(num_subjects):
    subject_sources = []
    for j in range(num_sources):
        amplitude = np.random.uniform(0.5, 1.5 )  # Random amplitude for each source
        frequency = np.random.uniform(0.05, 0.2)  # Random frequency for each source
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase for each source
        waveform = amplitude * np.sin(2 * np.pi * frequency * np.arange(num_samples) + phase)
        subject_sources.append(waveform)
    sources.append(np.array(subject_sources))


sources = np.array(sources).T
# Mix the sources for each subject to obtain observed mixtures
mixtures = np.array([np.dot(sources[:,:,i],mixing_matrices[i].T) for i in range(num_subjects)])



# Perform PCA on the observed mixtures
# make PCA for each subject

# Perform PCA on the observed mixtures for each subject
S_pca = np.zeros((num_subjects, num_samples, num_sources))
A_pca = np.zeros((num_subjects, num_sources, num_sources))
X_pca = np.zeros((num_subjects, num_samples, num_sources))
mu_pca = np.zeros((num_subjects, num_sources))
for i in range(num_subjects):
    pca = PCA(n_components=None, whiten=False)
    X_pca[i] = pca.fit(mixtures[i]).transform(mixtures[i])
    #S_pca[i] = pca.fit_transform(mixtures[i])
    A_pca[i] = pca.components_.T  # Get estimated mixing matrix
    S_pca[i] = np.dot(S_pca[i], A_pca[i].T)  # Estimate the sources
    mu_pca[i] = pca.mean_  # Get the mean of the mixtures
    assert np.allclose(mixtures[i], np.dot(X_pca[i], A_pca[i].T)+mu_pca[i])


#concatenate the mixtures
X_pca = np.concatenate(X_pca, axis=1)
A_pca2 = np.concatenate(A_pca, axis=1)
mixtures2 = np.concatenate(mixtures, axis=1)
mu_pca2 = np.concatenate(mu_pca, axis=0)
#assert np.allclose(mixtures2, X_pca @ A_pca2.T + mu_pca2)

# make whitening pca
# Perform ICA on the PCA-transformed mixtures
ica = FastICA(n_components=None, whiten='arbitrary-variance')
estimated_sources = ica.fit_transform(X_pca)

#get whitening matrix
whitening_matrix = ica.whitening_


# get estimated mixing matrices
estimated_mixing_matrices = ica.mixing_
# get A
estimated_A = ica.components_


assert np.allclose(X_pca, np.dot(estimated_sources, estimated_mixing_matrices.T)+ica.mean_)
print("Success!")



# print all the shapes please
print(np.shape(estimated_sources), "estimated sources")
print(np.shape(estimated_mixing_matrices), "estimated mixing matrices")
print(np.shape(A_pca[0]), "A_pca")
print(np.shape(whitening_matrix), "whitening matrix")
print(np.shape(estimated_A), "estimated A")


# get the mean from ica whitening matrix
mu_whitening = ica.mean_

"""
#backproject to normal space
back_proj_sub0 = A_pca[0] @ whitening_matrix[:3,:] @ (mu_whitening.reshape(9,1) + estimated_mixing_matrices @ estimated_sources.T )
back_proj_sub1 = A_pca[1] @ whitening_matrix[3:6:,:] @ (mu_whitening.reshape(9,1) + estimated_mixing_matrices @ estimated_sources.T)
back_proj_sub2 = A_pca[2] @ whitening_matrix[6:,:] @ (mu_whitening.reshape(9,1) + estimated_mixing_matrices @ estimated_sources.T)


# make backprojected sources a 3d matix
back_proj_sources = np.zeros((num_subjects, num_sources, num_samples))
back_proj_sources[0] = back_proj_sub0
back_proj_sources[1] = back_proj_sub1
back_proj_sources[2] = back_proj_sub2
"""

back_proj_sub0 = (X_pca[:,:3] @ A_pca[0].T) + mu_pca[0].reshape(1, 3)
back_proj_sub1 = (X_pca[:,3:] @ A_pca[1].T) + mu_pca[1].reshape(1, 3)


back_proj_sources = np.zeros((num_subjects,num_sources,num_samples))
back_proj_sources[0] = back_proj_sub0.T
back_proj_sources[1] = back_proj_sub1.T


print(np.shape(back_proj_sources), "back projected sources")

#assert np.allclose(mixtures[0], X_pca[:,:3] @ A_pca[0] + mu_pca[0])




# Plotting
num_plots = num_subjects
plt.figure(figsize=(10, 12))

# Plot the true sources for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
plt.suptitle("True Sources")

for i in range(num_plots):
    plt.subplot(num_plots, 1, i+1)
    for j in range(num_sources):
        plt.plot(sources[:,j ,i], label=f"Subject {i+1}, Source {j+1}")
    plt.legend()

plt.subplots_adjust(hspace=0.6)
plt.show()

# get ready to make a new plot
# data mixed signals in normal space


# Plot the observed mixtures for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
for i, ax in enumerate(axs):
    ax.set_title("Observed Mixtures")
    for j in range(num_mixtures):
        ax.plot(mixtures[i,: ,j], label=f"Subject {i+1}, Mixture {j+1}")
    ax.legend()
plt.subplots_adjust(hspace=0.6)
plt.show()




# Plot the estimated sources in ICA space
plt.figure(figsize=(10, 12))
plt.subplot(1, 1, 1)
plt.title("Estimated Sources")
for j in range(num_sources):
    plt.plot(np.arange(num_samples), estimated_sources[:,j], label=f"Source {j + 1}")
plt.legend()
plt.xlabel("Sample Index")
plt.subplots_adjust(hspace=0.6)
plt.show()
"""


#plot the estimated source for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
plt.suptitle("Estimated Sources")
for i, ax in enumerate(axs):
    ax.set_title(f"Subject {i+1}")
    for j in range(num_sources):
        ax.plot(np.arange(num_samples), estimated_sources_split[i][j], label=f"Source {j+1}")
    ax.legend()

plt.subplots_adjust(hspace=0.6)
plt.show()

"""

# plot backprojected sources
plt.figure(figsize=(10, 12))

fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
for i, ax in enumerate(axs):
    ax.set_title("back Mixtures")
    for j in range(num_mixtures):
        ax.plot(back_proj_sources[i,j,:], label=f"Subject {i+1}, Mixture {j+1}")
    ax.legend()
plt.subplots_adjust(hspace=0.6)
plt.show()
