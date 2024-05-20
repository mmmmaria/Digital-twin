# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:20:09 2023

@author: mariamegia
"""



import numpy as np
#from skimage import measure
from skimage.metrics import structural_similarity as ssim
#from skimage.measure import compare_msssim
from backend import import_excel
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

#https://pypi.org/project/sewar/
#https://sewar.readthedocs.io/en/latest/
#sewar.full_ref.msssim(GT, P, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None)[source]


X_train = import_excel('Dataset/X_train')
y_train = import_excel('Dataset/y_train')
X_test = import_excel('Dataset/X_test')
y_test = import_excel('Dataset/y_test')

x1=y_test

df2 = pd.read_excel("X_generated.xlsx")
dataset2 = df2.values
n_instance2=len(dataset2)
x2 = dataset2[:n_instance2,1]


# MS-SSIM
# the Multi-Scale Structural Similarity Index Measure (MS-SSIM) between two probability distributions 
# of 1-D data with unknown distribution

# Calculate the MS-SSIM between the two probability distributions

# msssim, _ = ssim(x1.reshape(1, 1, -1), x2.reshape(1, 1, -1), win_size=2, multichannel=False, gaussian_weights=True)
# print("MS-SSIM:", msssim) #winsize=7,11...

#Maximum Mean Discrepancy (MMD) for 1d data

def mmd(x1, x2, gamma):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between two sets of 1D data.
    
    Args:
    x1: numpy array of shape (n_samples_x,) containing the first set of 1D data.
    x2: numpy array of shape (n_samples_y,) containing the second set of 1D data.
    gamma: parameter for the Gaussian kernel used in the MMD calculation.
    
    Returns:
    mmd: float, the MMD between the two sets of 1D data.
    """
    n = x1.shape[0]
    m = x2.shape[0]
    Kxx = rbf_kernel(x1.reshape(-1, 1), x1.reshape(-1, 1), gamma=gamma)
    Kyy = rbf_kernel(x2.reshape(-1, 1), x2.reshape(-1, 1), gamma=gamma)
    Kxy = rbf_kernel(x1.reshape(-1, 1), x2.reshape(-1, 1), gamma=gamma)
    mmd = (1/n**2)*np.sum(Kxx) + (1/m**2)*np.sum(Kyy) - (2/(n*m))*np.sum(Kxy)
    return mmd

gamma = 1.0
mmd_value = mmd(x1, x2, gamma)
print(f"MMD value: {mmd_value}")

