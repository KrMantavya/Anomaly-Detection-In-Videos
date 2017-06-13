import os
import numpy
import PIL
from PIL import Image
import pickle

# Assuming all images are the same size, get dimensions of first image
w=256
h=256
N=200*30

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)

# Access all PNG files in directory
for i in range(1,31):
    allfiles=os.listdir("/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/train/train"+"%03d"%i+"/")
    imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg"]]

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imm=Image.open("/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/train/train"+"%03d"%i+"/"+im)
        imm=imm.resize((256,256),PIL.Image.ANTIALIAS)
        imarr=numpy.array(imm,dtype=numpy.float)
        arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("Average.jpg")
out.show()

# Save Mean Image for caffenet model
print arr.shape
arr= numpy.transpose(arr,(2,0,1))
numpy.save('MeanImage.npy',arr)
narr=numpy.load('/home/mantavya294/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
print arr.shape
print narr.shape
