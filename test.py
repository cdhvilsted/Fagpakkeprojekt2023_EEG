import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import random

# Set the random seed to 42
np.random.seed(42)

# make random seed to
random.seed(42)

# Generate simulated signals
num_subjects = 2  # Number of subjects
num_samples = 1000  # Number of samples per subject
num_sources = 3  # Number of independent sources per subject
num_mixtures = num_sources  # Number of observed mixtures per subject
common_source_position = 1
# Create random mixing matrices for each subject
mixing_matrices = [np.random.rand(num_mixtures, num_sources) for _ in range(num_subjects)]

"""
# Generate random independent sources for each subject
sources = []
common_source_position = random.randint(0, num_sources-1)  # Choose a random position for the common source
common_source = np.random.rand(num_samples)  # Generate a common source waveform

for i in range(num_subjects):
    subject_sources = []
    for j in range(num_sources):
        if j == common_source_position:
            waveform = common_source
        else:
            amplitude = np.random.uniform(20, 40)  # Random amplitude for each source
            frequency = np.random.uniform(0, 1)  # Random frequency for each source
            phase = np.random.uniform(0, 200 * np.pi)  # Random phase for each source
            waveform = amplitude * np.sin(2 * np.pi * frequency * np.arange(num_samples) + phase)
        subject_sources.append(waveform)
    sources.append(np.array(subject_sources))
"""
N = 1000
time = np.linspace(0, 10, N)

S1 = np.column_stack([np.sin(1.3*10 * np.pi * time),
                     np.cos(0.50* np.pi * time),
                     np.sin(1.1*10* np.pi * time)*3])
S2 = np.column_stack([np.sin(1.10*10 * np.pi * time)*3,
                     np.cos(0.50* np.pi * time),
                     np.sin(1.9*10 * np.pi * time)])

sources = np.zeros((2,3,1000))
sources[0] = S1.T

sources[1] = S2.T

sources = np.array(sources).T

# Mix the sources for each subject to obtain observed mixtures
mixtures = np.array([np.dot(sources[:, :, i], mixing_matrices[i].T) for i in range(num_subjects)])

# Perform PCA on the observed mixtures
S_pca = np.zeros((num_subjects, num_samples, num_sources))
A_pca = np.zeros((num_subjects, num_sources, num_sources))
X_pca = np.zeros((num_subjects, num_samples, num_sources))
mu_pca = np.zeros((num_subjects, num_sources))
for i in range(num_subjects):
    pca = PCA(n_components=None, whiten=False)
    X_pca[i] = pca.fit(mixtures[i]).transform(mixtures[i])
    A_pca[i] = pca.components_.T  # Get estimated mixing matrix
    S_pca[i] = np.dot(S_pca[i], A_pca[i].T)  # Estimate the sources
    mu_pca[i] = pca.mean_  # Get the mean of the mixtures
    assert np.allclose(mixtures[i], np.dot(X_pca[i], A_pca[i].T) + mu_pca[i])

# concatenate the mixtures
X_pca = np.concatenate(X_pca, axis=1)
A_pca2 = np.concatenate(A_pca, axis=1)
mixtures2 = np.concatenate(mixtures, axis=1)
mu_pca2 = np.concatenate(mu_pca, axis=0)

# Perform ICA on the PCA-transformed mixtures
ica = FastICA(n_components=None, whiten='arbitrary-variance', max_iter=1000, tol=0.0000001)
estimated_sources = ica.fit_transform(X_pca)

# get estimated mixing matrices
estimated_mixing_matrices = ica.mixing_
estimated_A = ica.components_



# Plotting
num_plots = num_subjects
plt.figure(figsize=(10, 12))

# Plot the true sources for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
plt.suptitle("True Sources")

for i in range(num_plots):
    plt.subplot(num_plots, 1, i + 1)
    for j in range(num_sources):
        if j == common_source_position:
            plt.plot(sources[:, j, i], label=f"Subject {i + 1}, Common Source {j + 1}", linestyle='--')
        else:
            plt.plot(sources[:, j, i], label=f"Subject {i + 1}, Source {j + 1}")
    plt.legend(loc='upper right')
    #plt.ylim(-1.5, 1.5)

plt.subplots_adjust(hspace=0.6)
plt.show()

# Plotting
num_plots = num_subjects
plt.figure(figsize=(10, 12))

# Plot the true sources for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
plt.suptitle("True Sources")

for i in range(num_plots):
    plt.subplot(num_plots, 1, i + 1)

    plt.plot(sources[:, 1, i], label=f"Subject {i + 1}, Common Source {1 + 1}", linestyle='--')
    plt.legend(loc='upper right')
plt.subplots_adjust(hspace=0.6)
plt.show()



# Plot the observed mixtures for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
for i, ax in enumerate(axs):
    ax.set_title("Observed Mixtures")
    for j in range(num_mixtures):
        ax.plot(mixtures[i, :, j], label=f"Subject {i + 1}, Mixture {j + 1}")
    ax.legend(loc='upper right')
plt.subplots_adjust(hspace=0.6)
plt.show()

# Plot the estimated sources in ICA space
plt.figure(figsize=(10, 12))
plt.subplot(2, 1, 1)
plt.title("Estimated Sources")
for j in range(num_sources*num_subjects):
    if j == common_source_position:
        plt.plot(np.arange(num_samples), estimated_sources[:, j], label=f"Estimated Common Source {j + 1}", linestyle='--')
    else:
        plt.plot(np.arange(num_samples), estimated_sources[:, j], label=f"Estimated Source {j + 1}")
plt.legend(loc='upper right')
plt.xlabel("Sample Index")
#make y-axis as (-1, 1)
#plt.ylim(-1, 1)
plt.subplots_adjust(hspace=0.6)
plt.show()



# Plot the estimated sources in ICA space
plt.figure(figsize=(10, 12))
plt.subplot(2, 1, 1)
plt.title("Estimated Sources component 3")
plt.plot(np.arange(num_samples), estimated_sources[:, 3], label=f"Estimated Common Source {4 + 1}", linestyle='-', color='red')
plt.legend(loc='upper right')
plt.xlabel("Sample Index")


#make y-axis as (-1, 1)
#plt.ylim(-1, 1)
plt.subplots_adjust(hspace=0.6)


# make a moving average of the estimated source 3
from scipy.ndimage import gaussian_filter1d
window_size = 20

moving_avg = gaussian_filter1d(estimated_sources[:, 3], window_size, mode = 'nearest')

plt.subplot(2, 1, 2)
plt.title("Moving average of estimated source component 3")
plt.plot(np.arange(num_samples), moving_avg, label=f"Estimated Common Source {4 + 1}", linestyle='--', color='red')
plt.legend(loc='upper right')
plt.xlabel("Sample Index")
#make y-axis as (-1, 1)
#plt.ylim(-1, 1)
plt.subplots_adjust(hspace=0.6)
plt.show()


# Plot backprojected sources
back_proj_sub0 = (X_pca[:, :3] @ A_pca[0].T) + mu_pca[0].reshape(1, 3)
back_proj_sub1 = (X_pca[:, 3:] @ A_pca[1].T) + mu_pca[1].reshape(1, 3)

back_proj_sources = np.zeros((num_subjects, num_sources, num_samples))
back_proj_sources[0] = back_proj_sub0.T
back_proj_sources[1] = back_proj_sub1.T

plt.figure(figsize=(10, 12))
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 12))
for i, ax in enumerate(axs):
    ax.set_title("Backprojected Mixtures")
    for j in range(num_mixtures):
        if j == common_source_position:
            ax.plot(back_proj_sources[i, j, :], label=f"Subject {i + 1}, Common Source {j + 1}", linestyle='--')
        else:
            ax.plot(back_proj_sources[i, j, :], label=f"Subject {i + 1}, Source {j + 1}")
    ax.legend(loc='upper right')
plt.subplots_adjust(hspace=0.6)
plt.show()

