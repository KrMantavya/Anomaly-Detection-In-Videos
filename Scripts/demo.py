import os
import sys
import AlexNetFeatureExtractor
import oneClassSVM
from sklearn.externals import joblib
import evaluate
import numpy as np
import featureReduction
import drawFigure
import pickle
sys.path.insert(0,'/home/mantavya294/caffe/python')
import caffe
import operator

clip = sys.argv[1]
clip=str(clip)

def getFeatures():
    modelDefinition='/home/mantavya294/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
    modelWeights='/home/mantavya294/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    frameCount=200
    features=np.zeros(shape=(frameCount,4096))

    print 'inside'
    #for i in range(1,frames+1):
    inputImageFile='/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'+'test'+'/'+clip+'/'+'{}'+'.jpg'
    net=caffe.Net(modelDefinition,modelWeights,caffe.TEST)

    #imageMeanFile='/home/mantavya294/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    imageMeanFile='/home/mantavya294/BTP/Scripts/MeanImage.npy'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(imageMeanFile).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(50,3,227,227)

    for k in range(4):
        for i in range(1,51):
            img=caffe.io.load_image(inputImageFile.format("%03d"%(k*50+i)))
            net.blobs['data'].data[(i-1),...]=transformer.preprocess('data',img)

        output=net.forward()

        for i in range(50):
            features[k*50+(i)]=net.blobs['fc6'].data[i][0]
    #print features
    #with open('/home/mantavya294/featureVector', 'w') as f:
    #    np.savetxt(f, net.blobs['fc8'].data[0], fmt='%.4f', delimiter='\n')
    return features


model=joblib.load("/home/mantavya294/BTP/Classifier/OCrbfSVM.pkl")


features=getFeatures()





distance=oneClassSVM.getDistance(model,features)

for d in range(10):
    dd=[]
    for p in range(20):
        dd.append(distance[d*20+p][0])
    min_index, min_value = min(enumerate(dd), key=operator.itemgetter(1))
    print min_index+20*d
    print min_value
