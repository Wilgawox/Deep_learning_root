import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import keras
import yaml
import tensorflow.keras.layers as tfk 
from keras.models import load_model
from keras.models import Sequential
import argparse
from keras.callbacks import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load paths.yml, need to change this to accept the user's YAML file if given
with open("paths.yml", 'r') as stream:
    paths = yaml.safe_load(stream)


class DataGenerator(keras.utils.Sequence):
    '''
    Class for creating keras dataset.
    '''

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=paths['batch_size'], dim=(512, 512), n_channels = 1, 
                 n_classes=2, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        Y = np.empty((self.batch_size,  *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp) :
            X[i,] = np.load(ID)
            Y[i,] = np.load(self.labels[ID]).astype(int)
        return X, Y


def create_partition(n_img, time, tile_number) :
    '''
    Generate dictionnaries containing the paths of the images stocked locally
    '''
    X_train = []
    X_val =  []
    X_test = []
    Y_path = {}
        
    for i in range(n_img) : 
        for t in range(time) :
            for n in range(tile_number) :
                if(i<10) : strI="000"+str(i+1)
                else :
                    if(i<100) : strI="00"+str(i+1)
                    else :
                        if(i<1000) : strI="0"+str(i+1)
                        else : strI=""+str(i+1)
                        
                #a = paths['dataset_path']+'ML1_input_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                #b = paths['dataset_path']+'ML1_result_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                
                path_to_x = data_path+'ML1_input_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                path_to_y = data_path+'ML1_result_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                
                if (i)*tile_number*time+tile_number*t+n>=(n_img*time*tile_number)/2 : X_train.append(path_to_x) # Put paths for percentage
                elif (i)*tile_number*time+tile_number*t+n<=(n_img*time*tile_number)/4 : X_val.append(path_to_x) # Put paths for percentage
                else : X_test.append(path_to_x)
                Y_path.update({path_to_x : path_to_y})
    img_path = {'train':tuple(X_train), 'val':tuple(X_val), 'test':tuple(X_test)}
    return img_path, Y_path