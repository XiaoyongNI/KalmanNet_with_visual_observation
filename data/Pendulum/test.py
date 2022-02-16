import numpy as np
test=np.load('pdl_data2.npz',encoding = "latin1")  
# print(test.dtype)
print(test.files)
print(test['Angle'].shape)
print(test['Location'].shape)
print(test['NoisyLoc'].shape)