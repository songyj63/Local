#%%
import numpy as np

data = np.loadtxt('C:/Users/VUNO/Desktop/Programming/Git/Study/Study01/Chapter10/data/housing.data.txt')

X = data[:,0:13]
Y = data[:,13]

print(Y)
print(X[0,:])

# %%
