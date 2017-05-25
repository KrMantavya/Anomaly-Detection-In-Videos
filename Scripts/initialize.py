import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM

print '\n GETTING TRAINING FEATURES'
folders=raw_input("number of folders")
frames=raw_input("number of frames")
folders=int(folders)
frames=int(frames)
features=AlexNetFeatureExtractor.getFeatures('Train',folders,frames)
print features.shape
y=raw_input('continue')

print '\n TRAINING THE CLASSIFIER'
model=oneClassSVM.getTrainedSVM(features)

print '\n GETTING TESTING FEATURES'
features=AlexNetFeatureExtractor.getFeatures('Test',folders,frames)
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
