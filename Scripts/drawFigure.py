import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

def plot3DView(features,prediction):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    pca=decomposition.PCA(n_components=3)
    pca.fit(features)
    reducedFeatures=pca.transform(features)

    for name, label in [('Normal', 1), ('Anomalous',-1)]:
        ax.text3D(features[prediction == label, 0].mean(),
              features[prediction == label, 1].mean() + 1.5,
              features[prediction == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    #cy = np.choose(y, [1, -1]).astype(np.float)
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=prediction, cmap=plt.cm.spectral)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()
    return
