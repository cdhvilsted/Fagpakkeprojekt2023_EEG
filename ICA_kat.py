import sklearn as sk
import mne
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import torch

randomSeed = random.seed(1)

directory = os.path.dirname(os.path.abspath(__file__))+"/EEGproj-main/EEGproj-main/data_preproc"

print("Working directory:  ", directory[:30],"/.../",directory[-57:], " ") # just printing the path to the data

Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15",
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]

Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19",
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]
Speech_files = [ i + "_4adj.set" for i in Speech]
Non_speech_files = [i + "_4adj.set" for i in Non_speech]
common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']

Aud_event = ["Tagi_A", "Tabi_A"] #only auditive

Vis_event = ["Tagi_V","Tabi_V"] #only visual

aud1person = None
file = Speech_files[1]
path = directory +"/"+ str(file)
path = path.replace(" ","")
raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
montage = raw.get_montage()
raw.pick_channels(common)
raw = raw.crop(tmin=-0.1)
print(raw.info)
event1 = raw[Aud_event[0]] # tagi
event2 = raw[Aud_event[1]] # tabi
aud1person = mne.concatenate_epochs([event1, event2])
aud1person=aud1person.drop([i for i in range(130,len(aud1person))])
aud1person = aud1person.get_data()
aud1person = np.swapaxes(aud1person, 1,2).reshape(36,-1)


liste = np.array(aud1person)
print(np.shape(liste))
for i in range(1,len(Speech_files)):
    file = Speech_files[i]
    path = directory + '/' + str(file)
    path = path.replace(" ","")
    raw = mne.io.read_epochs_eeglab(path, montage_units='dm')
    raw.pick_channels(common)
    raw = raw.crop(tmin=-0.1)
    event1 = raw[Aud_event[0]] # tagi
    event2 = raw[Aud_event[1]] # tabi 
    aud1person = mne.concatenate_epochs([event1, event2])
    aud1person=aud1person.drop([i for i in range(130,len(aud1person))])
    aud1person = aud1person.get_data()
    aud1person = np.swapaxes(aud1person, 1,2).reshape(36,-1)
    print('Aud1person shape is:', np.shape(aud1person))
    liste = np.dstack((liste,aud1person)) #hstack giver (trials, personer*antal kanaler, timesteps)


liste = np.swapaxes(liste,0,1)
liste = np.swapaxes(liste, 0,2)

print("shape:",np.shape(liste[0]))
print(np.shape(liste))

'''
# Create MNE Epochs object
reject = dict(mag=5e-12, grad=4000e-13)


# Apply ICA to the Epochs object
ica = mne.preprocessing.ICA(n_components=28, method='fastica',random_state=1) # 14 components for each group
#ica.fit(mne.EpochsArray(data_matrix_s, mne.create_info(Speech, 128)))


#change data_matrix_ns to epochs

# Make sure that the shape is (n_epochs, n_channels, n_samples(time)) 
# and that n_channels and n_samples are consistent with the in
# -formation contained in the common variable that you use to cre
# -ate the info argument in mne.create_info(common, 512).

ica.fit(liste, reject = reject)
# make a for loop for i in range
print([ica.get_explained_variance_ratio(liste, components=i, ch_type='eeg') for i in range(0,28)])

ica.plot_properties(liste, picks=5)

# Plot the topographic maps of the independent components
ica.plot_components()
'''


import numpy as np
from picard import picard
from multiviewica_master.multiviewica.reduce_data import reduce_data
from sklearn.utils.extmath import randomized_svd

def groupica(
    X,
    n_components=None,
    dimension_reduction="pca",
    max_iter=1000,
    random_state=None,
    tol=1e-7,
    ortho=False,
    extended=False,
):
    """
    Performs PCA on concatenated data across groups (ex: subjects)
    and apply ICA on reduced data.
    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    dimension_reduction: str, optional
        if srm: use srm to reduce the data
        if pca: use group specific pca to reduce the data
    max_iter : int, optional
        Maximum number of iterations to perform
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.
    ortho: bool, optional
        If True, uses Picard-O. Otherwise, uses the standard Picard.
    extended: None or bool, optional
        If True, uses the extended algorithm to separate sub and
        super-Gaussian sources.
        By default, True if ortho == True, False otherwise.
    Returns
    -------
    P : np array of shape (n_groups, n_components, n_features)
        P is the projection matrix that projects data in reduced space
    W : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    S : np array of shape (n_components, n_samples)
        Estimated source
    See also
    --------
    permica
    multiviewica
    """
    G, X = reduce_data(
        X, n_components=n_components, dimension_reduction=dimension_reduction
    )

    n_pb, p, n = X.shape
    X_concat = np.vstack(X)
    U, S, V = randomized_svd(X_concat, n_components=p)
    X_reduced = np.diag(S).dot(V)
    U = np.split(U, n_pb, axis=0)
    K, W, S = picard(
        X_reduced,
        ortho=ortho,
        extended=extended,
        centering=False,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    scale = np.linalg.norm(S, axis=1)
    S = S / scale[:, None]
    W = np.array([S.dot(np.linalg.pinv(x)) for x in X])
    return G, W, S

R, X = reduce_data(
        liste, n_components=10, dimension_reduction="pca"
    )

G, W, S = groupica(X, n_components=10, dimension_reduction="pca", max_iter=1000, random_state=None, tol=1e-7, ortho=False, extended=False)
# der udføres PCA på data

#print(np.shape(P)) # projection matrix / full unmixing matrix
print(np.shape(W)) # unmixing matrix
print(np.shape(S))

#A = np.linalg.inv(W) # mixing matrix (laver data til dimensions reduceret data)
#print(np.shape(A))

print('G:', np.shape(G))
print('R:', np.shape(R))

print('liste:', np.shape(liste))
print('X', np.shape(X))
#print('X2', np.shape(X2))
print(np.shape(S[0,:]))

#Y = W@G@R
# make an empty array of shape(14,10,36)
back_Y = np.zeros((14,10,36))

for i in range(14):
    #y = Y[i,:,:]
    result = W[i,:,:] @ np.transpose(np.linalg.pinv(G[i,:,:])) @ np.transpose(np.linalg.pinv(R[i,:,:]))
    #result = np.linalg.solve(y, S[i,:])
    back_Y[i,:,:] = result

print(np.shape(back_Y))

# data for hver forsøgsperson kommer af at gange mixing matrix med source for hver forsøgsperson fx X0 = A[0,0,:] @ S[0,:]

# sorter efter varians i hver component
#print(A[0,0,:])
#var_S = np.var(S, axis=1)
#print(np.argsort(var_S)) # sort index from smallest to highest variance


biosemi_montage = mne.channels.make_standard_montage('standard_1020',head_size=0.15)
print(montage.ch_names)
to_drop_ch = list(set(montage.ch_names)-set(common))
print(len(to_drop_ch))


fig, ax = plt.subplots(14,10, figsize=(15,12))
#plt.subplots_adjust(hspace=0.5)
print(len(common))
axs = ax.ravel()
count = 0
for j in range(14):
    for i in range(10):
        # make back_Y a list
        data = np.ndarray.tolist(back_Y[j,i])
        df = pd.DataFrame([data],columns=common)
        df[to_drop_ch] = 0
        df = df*1e-6
        df = df.reindex(columns=montage.ch_names)
        info = mne.create_info(ch_names=montage.ch_names,sfreq=10,ch_types='eeg')
        comp1 = mne.EvokedArray(df.to_numpy().T,info)
        comp1.set_montage(montage)
        comp1 = comp1.drop_channels(to_drop_ch)
        comp1.plot_topomap(times=[0],axes=axs[count],colorbar=False,show=False)
        ax[j, i].set_title(' ')
        ax[0, i].set_ylabel('Subject ' + str(i))
        ax[j, 0].set_xlabel('Component ' + str(j))
        ax[j, 0].xaxis.set_label_position('top')
        print(axs[count])
        print(count)
        count += 1
#comp1.plot()
#common1 =[i for i in biosemi_montage if i in common]
#print(np.shape(comp1))
#comp1 = mne.Epochs(comp1)
#print(type(comp1))
#print(comp1)
#comp1.plot_topomap(times=[0],sphere='eeglab')
plt.show()