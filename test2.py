import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import random

# Set the random seed to 42
np.random.seed(30) # this seed works for finding a suitable mixing matrix

# make random seed to
random.seed(30)

# Generate simulated signals
num_subjects = 2  # Number of subjects
num_samples = 1000  # Number of samples per subject
num_sources = 3  # Number of independent sources per subject
num_mixtures = num_sources  # Number of observed mixtures per subject

# Create random mixing matrices A for each subject
A = [np.random.rand(num_mixtures, num_sources) for _ in range(num_subjects)]
A = 



time = np.linspace(0, 10, num_samples)

# subject 1 sources
S1 = np.column_stack([np.sin(1.3*10 * np.pi * time),
                     np.sin(1.1*3* np.pi * time)*3,
                     np.cos(0.50* np.pi * time)])
# subject 1 sources
S2 = np.column_stack([np.cos(0.50* np.pi * time),
                     np.sin(1.1*10 * np.pi * time)*3,
                     np.sin(1.9*10 * np.pi * time)])

# making the source matrix S
S = np.zeros((2, 3, 1000))
S[0] = S1.T

S[1] = S2.T

S = np.array(S).T

# Mix the sources for each subject to obtain mixtured signal X
X = np.array([np.dot(S[:, :, i], A[i].T) for i in range(num_subjects)])

# Perform PCA on the observed mixtures
U_pca = np.zeros((num_subjects, num_samples, num_sources))
V_pca = np.zeros((num_subjects, num_sources, num_sources))
X_pca = np.zeros((num_subjects, num_samples, num_sources))
mu_pca = np.zeros((num_subjects, num_sources))

for i in range(num_subjects):
    pca = PCA(n_components=None, whiten=False)
    X_pca[i] = pca.fit(X[i]).transform(X[i]) # X in pca space
    V_pca[i] = pca.components_.T  # Mixing matrix estimated by PCA
    U_pca[i] = np.dot(U_pca[i], V_pca[i].T)  # Sources esitmated by PCA
    mu_pca[i] = pca.mean_  # Get the mean of the mixtures
    assert np.allclose(X[i], np.dot(X_pca[i], V_pca[i].T) + mu_pca[i]) # Check that the mixing matrix is correct


# concatenate the mixtures to prepare for ICA
X_pca = np.concatenate(X_pca, axis=1)


# Perform ICA on the PCA-transformed mixtures
ica = FastICA(n_components=None, whiten='arbitrary-variance', max_iter=10000, tol=0.0000001)
S_ICA = ica.fit_transform(X_pca)

#get whitening matrix
whitening_matrix = ica.whitening_


# get estimated mixing matrices
estimated_mixing_matrices = ica.mixing_
# get A
estimated_A = ica.components_



#assert np.allclose(X_pca, np.dot(S_ICA, estimated_mixing_matrices.T) @ whitening_matrix +ica.mean_, atol=0.1)
print("Success!")
#assert np.allclose(X_pca, np.dot(S_ICA, estimated_mixing_matrices.T) @ whitening_matrix +ica.mean_, atol=0.1)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Set seaborn plot style
sns.set(style='ticks', font_scale=1.2)

# Plotting
num_plots = num_subjects
plt.figure(figsize=(10, 11))

# Plot the true sources for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 8))
plt.suptitle("True Sources", fontsize=20, weight='bold')

for i in range(num_plots):
    plt.subplot(num_plots, 1, i + 1)
    for j in range(num_sources):
        if j == 2 and i == 0 or j == 0 and i == 1:
            plt.plot(S[:, j, i], label=f"Subject {i + 1}, Source {j + 1} (Common)", linewidth=2.5)
        else:
            plt.plot(S[:, j, i], label=f"Subject {i + 1}, Source {j + 1}", alpha=0.8)
    plt.legend(loc='upper right', fontsize='small')
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)

plt.subplots_adjust(hspace=0.6)
sns.despine() # Remove top and right spines from plots that share the same y-axis
plt.show()


# Plotting
num_plots = num_subjects
plt.figure(figsize=(10, 8))

# Plot the true sources for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 8))
plt.suptitle("Common True Sources", fontsize=20, weight='bold')

plt.subplot(num_plots, 1, 0 + 1)
plt.title("Subject 1", fontweight='bold')
plt.plot(S[:, 2, 0], label=f"Subject {0 + 1}, Source {2 + 1} (Common)", color='tab:green', linewidth=2.5)
plt.ylim(-3.3, 3.3)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(loc='upper right', fontsize='small')

plt.subplot(num_plots, 1, 1 + 1)
plt.title("Subject 2", fontweight='bold')
plt.plot(S[:, 0, 1], label=f"Subject {1 + 1}, Source {0 + 1} (Common)", color='tab:blue', linewidth=2.5)
plt.ylim(-3.3, 3.3)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(loc='upper right', fontsize='small')

plt.subplots_adjust(hspace=0.6)
sns.despine()
plt.show()


# Plot the observed mixtures for each subject
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 8))
plt.suptitle("Mixed Data", fontsize=20, weight='bold')

for i, ax in enumerate(axs):
    ax.set_title("Subject " + str(i + 1), fontweight='bold')
    for j in range(num_mixtures):
        ax.plot(X[i, :, j], label=f"Subject {i + 1}, Mixture {j + 1}", alpha=0.8)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)

plt.subplots_adjust(hspace=0.6)
sns.despine()
plt.show()


# Plot the estimated sources in ICA space
plt.figure(figsize=(10, 4))
plt.suptitle("Estimated Sources (ICA)", fontsize=20, weight='bold')

for j in range(num_sources * num_subjects):
    if j == 5:
        plt.plot(np.arange(num_samples), S_ICA[:, j], label=f"Estimated Source {j + 1} (Common)", linewidth=2.5, color = 'tab:pink')
    else:
        plt.plot(np.arange(num_samples), S_ICA[:, j], label=f"Estimated Source {j + 1}", alpha=0.8)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)

plt.legend(loc='upper right', fontsize='small')
plt.subplots_adjust(hspace=0.6)
sns.despine()
plt.show()


# Plot the estimated sources in ICA space
chosen_source = 5
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.suptitle("Estimated Source Component 2 (Common)", fontsize=20, weight='bold')

plt.plot(np.arange(num_samples), S_ICA[:, chosen_source], label=f"Estimated Source Component {1 + 1}",
         color='tab:pink', linewidth=2.5)
plt.legend(loc='upper right', fontsize='small')
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.subplots_adjust(hspace=0.6)

# Make a moving average of the estimated source 3
window_size = 24
moving_avg = gaussian_filter1d(S_ICA[:, chosen_source], window_size, mode='nearest')

plt.subplot(2, 1, 2)
plt.title("Moving Average of Estimated Source Component 2", fontweight='bold')
plt.plot(np.arange(num_samples), moving_avg, label=f"Moving Average", linestyle='--', color='tab:pink', linewidth=2)
plt.legend(loc='upper right', fontsize='small')
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.subplots_adjust(hspace=0.6)
sns.despine()
plt.show()


# Plot backprojected sources

back = np.dot(S_ICA, estimated_A)  + ica.mean_

back_proj_sub0 = (back[:, :3] @ V_pca[0].T) + mu_pca[0].reshape(1, 3)
back_proj_sub1 = (back[:, 3:] @ V_pca[1].T) + mu_pca[1].reshape(1, 3)

#back_proj_sub0 = (X_pca[:, :3] @ V_pca[0].T) + mu_pca[0].reshape(1, 3)
#back_proj_sub1 = (X_pca[:, 3:] @ V_pca[1].T) + mu_pca[1].reshape(1, 3)

back_proj_sources = np.zeros((num_subjects, num_sources, num_samples))
back_proj_sources[0] = back_proj_sub0.T
back_proj_sources[1] = back_proj_sub1.T

plt.figure(figsize=(10, 8))
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 8))
for i, ax in enumerate(axs):
    plt.suptitle("From ICA Backprojected to Mixed Data", fontsize=20, weight='bold')
    ax.set_title("Subject " + str(i + 1), fontweight='bold')

    for j in range(num_mixtures):
        ax.plot(back_proj_sources[i, j, :], label=f"Subject {i + 1}, Source {j + 1}", alpha=0.8)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)

plt.subplots_adjust(hspace=0.6)
sns.despine()
plt.show()
