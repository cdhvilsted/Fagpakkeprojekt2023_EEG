import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt



print((np.empty((13677,36))@np.empty((36,36))).shape)
"""
a = np.array([[[1e-5,1e-8,1],[2,4,2e-9]],[[2e-5,3e-8,4],[2,3,2e-10]]])
#print(np.min(a))
#print(round(np.min(a),3))

fig, axes = plt.subplots(4,14, figsize=(10, 3))

for r in range(4):
    for c in range(14):
        axes[r,c].imshow(np.random.rand(3,3))
        axes[r,c].set_xticks([])
        axes[r,c].set_yticks([])
        axes[r,c].set_title('')
        axes[0,c].set_title('test',fontsize=8)
        

for i in range(4):
    bbox1 = axes[i,0].get_position()
    bbox2 = axes[i,-1].get_position()
    rect = Rectangle((bbox1.xmin,bbox1.ymin),bbox2.xmax-bbox1.xmin,bbox2.ymax-bbox1.ymin,linewidth=1,edgecolor='grey',facecolor='none')
    fig.add_artist(rect)


for ax in axes[0,:]:
    bbox = ax.get_position()
    print(bbox)
    rect = Rectangle((bbox.xmin,bbox.ymin),bbox.width,bbox.height,linewidth=1,edgecolor='r',facecolor='none')
    fig.add_artist(rect)


plt.show()"""