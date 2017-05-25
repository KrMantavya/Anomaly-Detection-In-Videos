import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM
from sklearn.externals import joblib

print '\n GETTING TRAINING FEATURES'
folders=raw_input("number of folders")
folders=int(folders)
features=AlexNetFeatureExtractor.getFeatures('Train',folders)
print features.shape
y=raw_input('continue')

print '\n TRAINING THE CLASSIFIER'
model=oneClassSVM.getTrainedSVM(features)

print '\n Saving Model'
filename=raw_input("enter filename")
joblib.dump(model,'/home/mantavya294/BTP/Classifier/'+filename+'.pkl')
