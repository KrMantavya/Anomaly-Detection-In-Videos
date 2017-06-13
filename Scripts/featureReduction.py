import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

def applyPCA(features):
    pca=decomposition.PCA(n_components=512)
    pca.fit(features)
    reducedFeatures=pca.transform(features)
    return reducedFeatures
