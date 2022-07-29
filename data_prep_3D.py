import numpy as np
from PIL import ImageSequence


def create_Xarray(img_2Dt) : 
    # Create an array a sliced image
    X=[]
    for y in ImageSequence.Iterator(img_2Dt):
        X.append(np.array(y))
    return X


def create_Yarray(img_2D) : 
    #Create an array of the root growing as time pass
    img_2D=np.array(img_2D)
    Y=[]
    for t in range(1,int(np.max(img_2D)+1)) :
        temp=np.zeros((img_2D.shape[0], img_2D.shape[1]))
        for i in range(len(img_2D)) :
            for j in range(len(img_2D[:,])) :
                p=img_2D[i,j]
                if p!=0 and p<=t : 
                    temp[i,j]=p
        Y.append(temp)
    return Y

def create_Yarray_speedy(img_2D) :
    #Create an array of the root growing as time pass while reducing FOR loops
    img_2D=np.array(img_2D)
    Y=[]
    for t in range(1,int(np.max(img_2D)+1)) :
        maskA = img_2D <= t #Mask deleting the root parts in function of the time
        maskB = img_2D != 0 #Mask oscuring the pixels with value 0
        A = maskA*maskB
        Y.append(A)
    return np.array(Y)