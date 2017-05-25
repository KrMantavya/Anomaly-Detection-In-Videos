import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM
from sklearn.externals import joblib

print '\n LOADING THE MODEL'
filename=raw_input("enter filename")
model=joblib.load("/home/mantavya294/BTP/Classifier/"+filename+'.pkl')

print '\n GETTING TESTING FEATURES'
folders=raw_input("number of folders")
frames=raw_input("number of frames")
folders=int(folders)
frames=int(frames)
features=AlexNetFeatureExtractor.getValidationFeatures('Train',folders,frames)
print features.shape
y=raw_input('continue')

print '\n GETTING PREDICTION'
prediction=oneClassSVM.getPrediction(model,features)

print '\n GETTING DISTANCE'
distance=oneClassSVM.getDistance(model,features)

print '\n PREDICTIONS ARE \n'
print prediction

print '\n DISTANCES ARE'
print distance
