import os
import sys
from PIL import Image

for i in range(1,37):
    allfiles=os.listdir("/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test"+"%03d"%i+"/")
    imlist=[filename for filename in allfiles if  filename[-4:] in [".tif"]]

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imm=Image.open("/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test"+"%03d"%i+"/"+im)
        rgbimm=Image.new("RGBA",imm.size)
        rgbimm.paste(imm)
        rgbimm.save("/home/mantavya294/BTP/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/test/test"+"%03d"%i+"/"+im[0:3]+".jpg")
