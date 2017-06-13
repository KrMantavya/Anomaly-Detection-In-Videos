import os
import sys
from PIL import Image
import pickle
sys.path.insert(0,'/home/mantavya294/caffe/python')
import caffe

import numpy as np
from sklearn import utils
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getTrainedSVM(trainingData,nuu,g):
    model=svm.OneClassSVM(nu=nuu,kernel='rbf',gamma=g)
    #trainingData=np.reshape(trainingData,(1,len(trainingData)))
    model.fit(trainingData)
    return model

def getPrediction(model,testingData):
    #testingData=np.reshape(testingData,(1,len(testingData)))
    prediction=model.predict(testingData)
    return prediction

def getAccuracy(prediction,folder,datasetNo):

    if int(datasetNo)==1:
        labels = np.ones(shape=(36,200))
        labels[0][60:152]=-1
        labels[1][50:175]=-1
        labels[2][91:200]=-1
        labels[3][31:168]=-1
        labels[4][[range(5,90) + range(140,200)]]=-1
        labels[5][[range(1,100) + range(110,200)]]=-1
        labels[6][1:175]=-1
        labels[7][1:94]=-1
        labels[8][1:48]=-1
        labels[9][1:140]=-1
        labels[10][70:165]=-1
        labels[11][130:200]=-1
        labels[12][1:156]=-1
        labels[13][1:200]=-1
        labels[14][138:200]=-1
        labels[15][123:200]=-1
        labels[16][1:47]=-1
        labels[17][54:120]=-1
        labels[18][64:138]=-1
        labels[19][45:175]=-1
        labels[20][31:200]=-1
        labels[21][16:107]=-1
        labels[22][8:165]=-1
        labels[23][50:171]=-1
        labels[24][40:135]=-1
        labels[25][77:144]=-1
        labels[26][10:122]=-1
        labels[27][105:200]=-1
        labels[28][[range(1,15) +  range(45,113)]]=-1
        labels[29][175:200]=-1
        labels[30][1:180]=-1
        labels[31][[range(1,52) + range(65,115)]]=-1
        labels[32][5:165]=-1
        labels[33][1:121]=-1
        labels[34][86:200]=-1
        labels[35][15:108]=-1

    tp=0.0
    fp=0.0
    matches=0.0
    label=np.zeros(shape=(200*folder))
    for i in range(int(folder)):
        truePositive=0
        falsePositive=0
        ones=0
        for j in range(200):
            label[i*200+j]=labels[i][j]
            if labels[i][j]==1:
                ones+=1
                if prediction[i*200+j]==-1:
                    falsePositive+=1
                else:
                    matches+=1
            else:
                if prediction[i*200+j]==-1:
                    truePositive+=1
                    matches+=1


        if ones!=0:
            t_fp = float(falsePositive)/float(ones)
        else:
            t_fp=0.0



        t_tp = float(truePositive)/float(200-ones)

        tp+=t_tp
        fp+=t_fp
    accuracy=float(matches)/float(folder*200)
    tp/=float(folder)
    fp/=float(folder)
    return tp,fp,accuracy,label

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def getFeatures(purpose,folders):
    modelDefinition='/home/mantavya294/BTP/Models/cnn_model/deploy.prototxt'
    modelWeights='/home/mantavya294/BTP/Models/cnn_model/caffenet_train_iter_10000.caffemodel'
    frameCount=folders*200
    features=np.zeros(shape=(frameCount,40))
    for j in range(1,folders+1):
        print 'inside'
        #for i in range(1,frames+1):
        inputImageFile='/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'+purpose+'/'+purpose+"%03d"%j+'/'+'{}'+'.jpg'
        net=caffe.Net(modelDefinition,modelWeights,caffe.TEST)

        #imageMeanFile='/home/mantavya294/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
        imageMeanFile='/home/mantavya294/BTP/Scripts/MeanImage.npy'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(imageMeanFile).mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 255.0)

        net.blobs['data'].reshape(20,3,227,227)

        for k in range(10):
            for i in range(1,21):
                img=caffe.io.load_image(inputImageFile.format("%03d"%(k*20+i)))
                net.blobs['data'].data[(i-1),...]=transformer.preprocess('data',img)

            output=net.forward()

            for i in range(20):
                features[(j-1)*200+k*20+(i)]=net.blobs['fc8'].data[i][0]
    #print features
    #with open('/home/mantavya294/featureVector', 'w') as f:
    #    np.savetxt(f, net.blobs['fc8'].data[0], fmt='%.4f', delimiter='\n')
    return features


x=[]
y=[]
z=[]
#features=getFeatures('train',32)
#np.save('training.npy',features)
features=np.load('training.npy')
#feature=getFeatures('test',36)
#np.save('testing.npy',feature)
feature=np.load('testing.npy')
nuu=0.000001
while nuu<=0.00002:
    g=0.000001
    while g<=0.00002:
        model=getTrainedSVM(features,nuu,g)
        prediction=getPrediction(model,feature)
        #print prediction.shape
        tp,fp,accuracy,label=getAccuracy(prediction,36,1)
        x.append(nuu)
        y.append(g)
        z.append(accuracy*100)
        g+=0.000001
    nuu+=0.000001
print len(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axes = plt.gca()
zmin=min(z)
zmax=max(z)
print zmin
print '\n'
print zmax
axes.set_zlim([zmin,zmax])
ax.set_xlabel('nu')
ax.set_ylabel('gamma')
ax.set_zlabel('Overall Accuracy')
ax.plot(x, y, z, label = 'curve')
plt.show()
