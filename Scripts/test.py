import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM
from sklearn.externals import joblib
import evaluate
import numpy as np
import featureReduction
import drawFigure

print '\n LOADING THE MODEL'
filename=raw_input("enter filename")
model=joblib.load("/home/mantavya294/BTP/Classifier/"+filename+'.pkl')

print '\n GETTING TESTING FEATURES'
folders=raw_input("number of folders")
folders=int(folders)
#features=AlexNetFeatureExtractor.getFeatures('test',folders)
features=np.load('testingFeatures1.npy')
print features.shape
#np.save('testingFeatures1.npy',features)
y=raw_input('continue')

print '\n DEMENSIONALITY REDUCTION'
#features=featureReduction.applyPCA(features)

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
truePositive,falsePositive,accuracy,labels=evaluate.getAccuracy(prediction,folders,datasetNo)

print '\n PERCENTAGE OF FALSE POSITIVES ARE'
print falsePositive

print '\n ACCURACY IN ANOMALOUS FRAME DETECTION IS'
print truePositive

print '\n OVERALL ACCURACY IS'
print accuracy

#print '\n GETTING SPECTRAL MAP'
#drawFigure.plot3DView(features,labels)
#drawFigure.plot3DView(features,prediction)
