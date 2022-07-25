#Imports
from glob import glob
import numpy as np
from keras.callbacks import *
import paths


def create_IO_for_CNN_training(n_img, time, n_tile) :
    '''
    Parameters
    ----------
    n_img : int
        Number of different image we will load into the dataset
    time : int
        Number of slices of each 2d+t images
    n_tile : int
        Number of tiles from each slice

    Returns
    -------
    X : list
        List containing numpy array of the images that will be used to create the X array
    Y : list
        List containing numpy array of the images that will be used to create the Y array 
    '''

    X = []
    Y = []
    for i in range(1, n_img+1) : 
        for t in range(time) :
            for n in range(n_tile) :
                if(i<10) : strI="000"+str(i)
                else :
                    if(i<100) : strI="00"+str(i)
                    else :
                        if(i<1000) : strI="0"+str(i)
                        else : strI=""+str(i)
                a = np.load(paths.dataset_path+'ML1_input_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy')
                b = np.load(paths.dataset_path+'ML1_result_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy')
                X.append(a)
                Y.append(b)
    return X, Y


def shuffle_XY(list_X, list_Y, percent_train, percent_valid, percent_test) :
    '''
    Parameters
    ----------
    list_X, list_Y : list
        Number of different image we will load into the dataset
    percent_train : int
        Percentage of images going into the training dataset
    percent_valid : int
        Percentage of images going into the validation dataset
    percent_test : int
        Percentage of images going into the test dataset

    Returns
    -------
    X_train, Y_train : list
        Training dataset as a list
    X_train, Y_train : list
        Test dataset as a list
    X_train, Y_train : list
        Validation dataset as a list
    '''

    N = len(list_X)
    random_list = np.arange(N)
    np.random.shuffle(random_list)
    
    X_temp = np.array(list_X)[random_list]
    Y_temp = np.array(list_Y)[random_list]
    
    X_train = X_temp[0:(percent_train*N)//100]
    X_valid = X_temp[0:(percent_valid*N)//100]
    X_test = X_temp[0:(percent_test*N)//100]
    
    Y_train = Y_temp[0:(percent_train*N)//100]
    Y_valid = Y_temp[0:(percent_valid*N)//100]
    Y_test = Y_temp[0:(percent_test*N)//100]
    return X_train,Y_train,X_test,Y_test,X_valid,Y_valid