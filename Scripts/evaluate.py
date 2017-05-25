import os
import sys
import numpy as np

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

    for i in range(int(folder)):
        truePositive=0
        falsePositive=0
        ones=0
        for j in range(200):
            if labels[i][j]==1:
                ones+=1
                if prediction[i*200+j]==-1:
                    falsePositive+=1
            else:
                if prediction[i*200+j]==-1:
                    truePositive+=1

        print "\n PERCENTAGE OF FALSE POSITIVE"
        if ones!=0:
            t_fp = float(falsePositive)/float(ones)
        else:
            t_fp=0.0
        print t_fp

        print "\n TRUE POSITIVE ACCURACY"
        t_tp = float(truePositive)/float(200-ones)
        print t_tp

        tp+=t_tp
        fp+=t_fp
    tp/=float(folder)
    fp/=float(folder)
    return tp,fp
