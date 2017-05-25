import os
import sys
from PIL import Image
import numpy as np
import pickle
sys.path.insert(0,'/home/mantavya294/caffe/python')
import caffe

def getFeatures(purpose,folders):
    modelDefinition='/home/mantavya294/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
    modelWeights='/home/mantavya294/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    frameCount=folders*200
    features=np.zeros(shape=(frameCount,4096))
    for j in range(1,folders+1):
        print 'inside'
        #for i in range(1,frames+1):
        inputImageFile='/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'+purpose+'/'+purpose+"%03d"%j+'/'+'{}'+'.tif'
        net=caffe.Net(modelDefinition,modelWeights,caffe.TEST)

        imageMeanFile='/home/mantavya294/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

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
                features[(j-1)*200+k*20+(i)]=net.blobs['fc6'].data[i][0]
    #print features
    #with open('/home/mantavya294/featureVector', 'w') as f:
    #    np.savetxt(f, net.blobs['fc8'].data[0], fmt='%.4f', delimiter='\n')
    return features

def getValidationFeatures(purpose,folders,frames):
    modelDefinition='/home/mantavya294/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
    modelWeights='/home/mantavya294/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    frameCount=(36-folders)*frames
    features=np.zeros(shape=(frameCount,4096))
    for j in range(folders+1,37):
        print 'inside'
        for i in range(1,frames+1):
            inputImageFile='/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'+purpose+'/'+purpose+"%03d"%j+'/'+"%03d"%i+'.tif'
            net=caffe.Net(modelDefinition,modelWeights,caffe.TEST)

            imageMeanFile='/home/mantavya294/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_mean('data', np.load(imageMeanFile).mean(1).mean(1))
            transformer.set_transpose('data', (2,0,1))
            transformer.set_raw_scale('data', 255.0)

            net.blobs['data'].reshape(1,3,227,227)

            img=caffe.io.load_image(inputImageFile)

            net.blobs['data'].data[...]=transformer.preprocess('data',img)

            output=net.forward()
            features[(j-1)*frames+(i-1)]=net.blobs['fc6'].data[0]
    #print features
    #with open('/home/mantavya294/featureVector', 'w') as f:
    #    np.savetxt(f, net.blobs['fc8'].data[0], fmt='%.4f', delimiter='\n')
    return features
