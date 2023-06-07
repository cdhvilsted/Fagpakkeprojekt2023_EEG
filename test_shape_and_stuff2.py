import numpy as np

a = np.array([[[1e-5,1e-8,1],[2,4,2e-9]],[[2e-5,3e-8,4],[2,3,2e-10]]])
print(np.min(a))
print(round(np.min(a),3))