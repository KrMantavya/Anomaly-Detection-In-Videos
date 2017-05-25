import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM
from sklearn.externals import joblib
import evaluate

print '\n LOADING THE MODEL'
filename=raw_input("enter filename")
model=joblib.load("/home/mantavya294/BTP/Classifier/"+filename+'.pkl')

print '\n GETTING TESTING FEATURES'
folders=raw_input("number of folders")
folders=int(folders)
features=AlexNetFeatureExtractor.getFeatures('Test',folders)
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

print '\n GETTING ACCURACY'
datasetNo=raw_input("Enter Dataset No")
truePositive,falsePositive=evaluate.getAccuracy(prediction,datasetNo,folders)

print '\n PERCENTAGE OF FALSE POSITIVES ARE'
print falsePositive

print '\n ACCURACY IN ANOMALOUS FRAME DETECTION IS'
print truePositive
