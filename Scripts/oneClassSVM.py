import numpy as np
from sklearn import utils
from sklearn import svm
import matplotlib

def getTrainedSVM(trainingData):
    model=svm.OneClassSVM(nu=0.00005,kernel='rbf',gamma=0.0001)
    #trainingData=np.reshape(trainingData,(1,len(trainingData)))
    print trainingData.shape
    model.fit(trainingData)
    return model

def getPrediction(model,testingData):
    #testingData=np.reshape(testingData,(1,len(testingData)))
    prediction=model.predict(testingData)
    return prediction

def getDistance(model,testingData):
    distance=model.decision_function(testingData)
    return distance
