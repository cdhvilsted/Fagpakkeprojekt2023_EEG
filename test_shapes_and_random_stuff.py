import numpy as np
import mne
import os as os
from ICA_dataImport import common, data_As, data_Vs, data_As, data_AVcs, data_AVics, data_AVc, data_A, data_V, data_AVic
import matplotlib.pyplot as plt

"""
g1 = np.empty((140,10))
R1 = np.empty((10,36))

print(np.dot(g1,R1).shape)
print(np.linalg.pinv(R1).shape)
print(np.dot(np.linalg.pinv(R1),np.transpose(g1)).shape)
"""
"""
# hstack two matrices
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
c = np.vstack((a, b))
d = np.vstack((b, a))
e = np.array([[1,2],[3,4],[5,6]])
print(c)
print(np.transpose(c))
print(d)
print(np.transpose(a))
print(e)
print(np.transpose(e))

"""

#plot data for speech files
pdata_As = np.mean(data_As[0,12,:].reshape(97,141),axis=0)
pdata_Vs = np.mean(data_Vs[0,12,:].reshape(97,141),axis=0)
pdata_AVcs = np.mean(data_AVcs[0,12,:].reshape(97,141),axis=0)
pdata_AVics = np.mean(data_AVics[0,12,:].reshape(97,141),axis=0)

#plot data for non-speech files
pdata_AVc = np.mean(data_AVc[0,12,:].reshape(97,141),axis=0)
pdata_A = np.mean(data_A[0,12,:].reshape(97,141),axis=0)
pdata_V = np.mean(data_V[0,12,:].reshape(97,141),axis=0)
pdata_AVic = np.mean(data_AVic[0,12,:].reshape(97,141),axis=0)
for i in range(1,14):
    pdata_As = np.vstack((pdata_As,np.mean(data_As[i,12,:].reshape(97,141),axis=0)))
    pdata_Vs = np.vstack((pdata_Vs,np.mean(data_Vs[i,12,:].reshape(97,141),axis=0)))
    pdata_AVcs = np.vstack((pdata_AVcs,np.mean(data_AVcs[i,12,:].reshape(97,141),axis=0)))
    pdata_AVics = np.vstack((pdata_AVics,np.mean(data_AVics[i,12,:].reshape(97,141),axis=0)))
    pdata_AVc = np.vstack((pdata_AVc,np.mean(data_AVc[i,12,:].reshape(97,141),axis=0)))
    pdata_A = np.vstack((pdata_A,np.mean(data_A[i,12,:].reshape(97,141),axis=0)))
    pdata_V = np.vstack((pdata_V,np.mean(data_V[i,12,:].reshape(97,141),axis=0)))
    pdata_AVic = np.vstack((pdata_AVic,np.mean(data_AVic[i,12,:].reshape(97,141),axis=0)))

print(pdata_As.shape)
print(np.mean(pdata_As,axis=0).shape)
pdata_As = np.mean(pdata_As,axis=0)
pdata_Vs = np.mean(pdata_Vs,axis=0)
pdata_AVcs = np.mean(pdata_AVcs,axis=0) - pdata_Vs
pdata_AVics = np.mean(pdata_AVics,axis=0) - pdata_Vs

pdata_A = np.mean(pdata_A,axis=0)
pdata_V = np.mean(pdata_V,axis=0)
pdata_AVic = np.mean(pdata_AVic,axis=0) - pdata_V
pdata_AVc = np.mean(pdata_AVc,axis=0) - pdata_V


x = np.arange(-0.1,1,step=1/128)
fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.axvline(x=0.1,color='r')
ax1.plot(x,pdata_As,color='k', label='A')
ax1.plot(x,pdata_AVics,color='k',linestyle='dashed', label='Incongruent AV')
ax1.plot(x,pdata_AVcs,color='0.8', label = 'Congruent AV')
ax1.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax1.invert_yaxis()
ax1.set_title('Speech')
#ax2.axvline(x=0.1,color='r')
ax2.plot(x,pdata_A,color='k', label='A')
ax2.plot(x,pdata_AVic,color='k',linestyle='dashed', label='Incongruent AV')
ax2.plot(x,pdata_AVc,color='0.8', label = 'Congruent AV')
ax2.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax2.invert_yaxis()
ax2.set_title('Non-speech')
ax1.legend(loc='upper right', fontsize = 8)
ax2.legend(loc='upper right', fontsize = 8)
plt.show()
