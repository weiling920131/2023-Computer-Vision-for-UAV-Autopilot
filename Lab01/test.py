import numpy as np
a = np.array([[1,2],[3,4]])
a=np.insert(a, 1, values=np.zeros([1,2]),axis=0)
print(a)