import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM
from sklearn.externals import joblib
import numpy as np
import featureReduction

print '\n GETTING TRAINING FEATURES'
folders=raw_input("number of folders")
folders=int(folders)
#features=AlexNetFeatureExtractor.getFeatures('train',folders)
features=np.load('trainingFeatures1.npy')
print features.shape
#np.save('trainingFeatures1.npy',features)
y=raw_input('continue')

print '\n DEMENSIONALITY REDUCTION'
#features=featureReduction.applyPCA(features)

print '\n TRAINING THE CLASSIFIER'
model=oneClassSVM.getTrainedSVM(features)

print '\n Saving Model'
filename=raw_input("enter filename")
joblib.dump(model,'/home/mantavya294/BTP/Classifier/'+filename+'.pkl')
