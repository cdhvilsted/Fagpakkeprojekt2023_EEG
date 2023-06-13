from our_group_ICA import loadData
import numpy as np
import matplotlib.pyplot as plt

# SCRIPT FOR VISUALIZING RANDOM EPOCHS FROM TEST SUBJECT TOGEHTER WITH ERP
# Reveals how noise is averaged out in ERP's

data_A, data_V, data_AVc, data_AVic, data_As, data_Vs, data_AVcs, data_AVics = loadData()

# Auditive data
print(" ")
print("Shape of auditive data: ", data_A.shape)
print("Choosing subject number: ", 1)
# Choosing epochs for channel Cz
data_A1 = data_A[0,12,:].reshape(97,141)
print("Shape of auditive data for subject 1: ", data_A1.shape)
print(" ")

# Choosing 2 random epochs
epoch_1 = data_A1[33,:]
epoch_2 = data_A1[60,:]

# ERP
erp_A1 = np.mean(data_A1,axis=0)


# Audiovisual-congruent data
print(" ")
print("Shape of audiovisual-congruent data: ", data_AVc.shape)
print("Choosing subject number: ", 1)
# Choosing epochs for channel Cz
data_AVc1 = data_AVc[0,12,:].reshape(97,141)
data_V1 = data_V[0,12,:].reshape(97,141)
print("Shape of audiovisual-congruent data for subject 1: ", data_AVc1.shape)
print(" ")

# Choosing 2 random epochs
epoch_AVc1 = data_AVc1[20,:] - data_V1[20,:]
epoch_AVc2 = data_AVc1[-1,:] - data_V1[-1,:]

# ERP
erp_AVc1 = np.mean(data_AVc1,axis=0) - np.mean(data_V1, axis=0)


# Plotting
x = np.arange(-0.1,1,step=1/128)
fig, ax = plt.subplots(2,2)

ax[0,0].plot(x,epoch_1,color='k', label='Epoch 33 (A)')
ax[0,0].plot(x,epoch_2,color='0.4',linestyle='dashed', label='Epoch 60 (A)')

ax[1,0].plot(x,epoch_AVc1,color='k', label='Epoch 20 (AVc-V)')
ax[1,0].plot(x,epoch_AVc2,color='0.4',linestyle='dashed', label='Epoch 97 (AVc-V)')

ax[0,0].invert_yaxis()
ax[1,0].invert_yaxis()

ax[0,0].set_title('Non-speech epochs for subject 1')

ax[0,0].legend(loc='upper right', fontsize = 8)
ax[1,0].legend(loc='upper right', fontsize = 8)

ax[0,1].plot(x,erp_A1,color='k', label='ERP (A)')
ax[1,1].plot(x,erp_AVc1,color='k', label='ERP (AVc-V)')

ax[0,1].invert_yaxis()
ax[1,1].invert_yaxis()
ax[0,1].set_title('Non-speech ERP for subject 1')
ax[0,1].legend(loc='upper right', fontsize = 8)
ax[1,1].legend(loc='upper right', fontsize = 8)

plt.show()