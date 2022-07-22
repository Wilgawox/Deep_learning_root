import numpy as np
import tensorflow.keras.layers as tfk 
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt
try:
    from tensorflow.keras.utils.all_utils import Sequence as Seq
except ModuleNotFoundError as err:
    from tensorflow.keras.utils import Sequence as Seq
    pass


def get_the_data(labels,list_ID):
    indexes = np.arange(100) 
    indexes = indexes[1:100]

        # Find list of IDs
    list_IDs_temp = [list_ID[k] for k in indexes]

        # Generate data
    X = np.empty((100, [512,512]), dtype=float)
    Y = np.empty((100,[512,512] ), dtype=int)

    for i, ID in enumerate(list_IDs_temp) :
        X[i,] = np.load(ID)
        Y[i,] = np.load(labels[ID]).astype(int)
    return X,Y

class DataGenerator(Seq):
#class DataGenerator(keras.utils.all_utils.Sequence):
    '''
    Class for creating keras dataset.
    '''

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=16, dim=(512, 512), n_channels = 1, 
                 n_classes=2, shuffle=False,paths="paths.yml"):
        self.dim = dim
        self.batch_size =batch_size
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
        #print("STARTING DATA_GENERATION")
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        Y = np.empty((self.batch_size,  *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp) :
            X[i,] = np.load(ID)
            Y[i,] = np.load(self.labels[ID]).astype(int)

        list_of_non_null_tiles=[]
        nb_empty=0
        nb_full=0
        mean=0
        for i in range(len(Y)):
            su=np.sum(Y[i,])
            if(su==0):
                nb_empty=nb_empty+1
            else:
                #print(100.0*su/(512.0*512.0))
                nb_full=nb_full+1
                mean=mean+su
                list_of_non_null_tiles.append(i)
        mean=mean/(len(Y))
        #print("\nHere we generate a new batch. Over the batch Y, of len "+str(nb_full)+", the mean value of the non null Y's is "+str(mean))
        #print("And then we got : empty="+str(nb_empty)+" and full="+str(nb_full))
        #print("Thus, the dict stuff thing of indexes is")
        X=X[list_of_non_null_tiles]
        Y=Y[list_of_non_null_tiles]
        return X, Y


def create_partition(n_img, time, tile_number,paths) :
    '''
    Generate dictionnaries containing the paths of the images stocked locally
    '''
    X_train = []
    X_val =  []
    X_test = []
    Y_path = {}

    if(n_img<4):
        print("EN PRISON !")
        exit()    
    for i in range(1, n_img+1) : 
        for t in range(time) :
            for n in range(tile_number) :
                if(i<10) : strI="000"+str(i)
                else :
                    if(i<100) : strI="00"+str(i)
                    else :
                        if(i<1000) : strI="0"+str(i)
                        else : strI=""+str(i)
                
                path_to_x =  paths['dataset_path']+'ML1_input_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                path_to_y = paths['dataset_path']+'ML1_result_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                
                if (i)>=(n_img)/2 : X_train.append(path_to_x)
                elif (i)<=(n_img)/4 : X_val.append(path_to_x)
                else : X_test.append(path_to_x)
                Y_path.update({path_to_x : path_to_y})

    img_path = {'train':tuple(X_train), 'val':tuple(X_val), 'test':tuple(X_test)}

    return img_path, Y_path