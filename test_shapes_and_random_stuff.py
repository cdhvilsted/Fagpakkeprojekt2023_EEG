import numpy as np
import mne
import os as os
from ICA_dataImport import EEGdata, montage, common
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
EEGdata1 = EEGdata[0]
print(EEGdata1.shape)
print(common.index('Cz'))

data = np.empty((141,130))
print(EEGdata[0,12,:].shape)
for i in range(14):
    data = np.hstack((data,EEGdata[i,12,:].reshape(141,130)))
print(data.shape)
print(np.mean(data,axis=1).shape)
data = np.mean(data,axis=1)

x = np.arange(-0.1,1,step=1/128)
fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.axvline(x=0.1,color='r')
ax1.plot(x,data,color='k', label='A')
ax1.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax1.invert_yaxis()
ax1.set_title('Speech')
#ax2.axvline(x=0.1,color='r')
ax2.plot(x,data, color='k', label='A')
ax2.set_yticks(np.arange(-6e-6,8e-6,2e-6))
ax2.invert_yaxis()
ax2.set_title('Non-speech')
ax1.legend(loc='upper right', fontsize = 8)
ax2.legend(loc='upper right', fontsize = 8)
plt.show()
