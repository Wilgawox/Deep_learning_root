
print('START')

# Dataset creation based on : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

#tensorboard --logdir Documents/CIRAD_stage_2022/Deep_learning_root/logs

#python cnn_dataset.py CNN_dataset

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)

#import paths as p # TODO : put this in a separate file with variables
import ranging_and_tiling_helpers
import dataset_config
import dataset_creation
print('imported local files')

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import datetime
import yaml
import tensorflow.keras.layers as tfk
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#En entrée, de -inf a + inf
#En sortie, is lower ? 0 : 1 
def apply_threshold(y_pred):
    return K.round(y_pred)

def my_tp(y_true, y_pred):
    return K.sum(K.round(y_true * apply_threshold(y_pred)))

def my_tn(y_true, y_pred):
    return K.sum(K.cast(K.equal(K.round(y_true + apply_threshold(y_pred)), 0), K.floatx()))

def my_fp(y_true, y_pred):
    return K.sum(K.cast(K.equal(K.round(apply_threshold(y_pred)) - y_true, 1), K.floatx()))

def my_fn(y_true, y_pred):
    return K.sum(K.cast(K.equal(y_true - K.round(apply_threshold(y_pred)), 1), K.floatx()))

def my_pos(y_true, y_pred):
    return K.sum(K.round(y_true))

def my_neg(y_true, y_pred):
    return K.sum(K.round(1-y_true))

def my_precision(y_true, y_pred):
    return my_tp(y_true, y_pred) / (my_tp(y_true, y_pred) + my_fp(y_true, y_pred))

def my_recall(y_true, y_pred):
    return my_tp(y_true, y_pred) / (my_tp(y_true, y_pred) + my_fn(y_true, y_pred))

def my_precision2(y_true, y_pred):
    return (K.sum(K.round(y_true * apply_threshold(y_pred)))) / (  (K.sum(K.round(y_true * apply_threshold(y_pred)))) + (K.sum(K.cast(K.equal(K.round(apply_threshold(y_pred)) - y_true, 1), K.floatx()))))

def my_recall2(y_true, y_pred):
    return my_tp(y_true, y_pred) / (my_tp(y_true, y_pred) + my_fn(y_true, y_pred))




def my_loss(y_true, y_pred, sample_weight=[1,1]):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    print(y_true)
#    return bce(y_true, y_pred)
    output=tf.convert_to_tensor(y_pred)
    epsilon_=tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

    # Compute cross entropy from probabilities.
    target=tf.convert_to_tensor(y_true)
    bce = K.cast(sample_weight[1],K.floatx()) * K.cast(tf.math.log(output + K.epsilon()),K.floatx())
    bce += K.cast(sample_weight[0],K.floatx()) * K.cast(1 - target,K.floatx()) * K.cast(tf.math.log(1 - output + K.epsilon()),K.floatx())
    return -bce

def CNN_dataset(args) : 
    #############################################################
    ####### Create partition of data and data generators ########
    #############################################################
    ###              .'',                           .'',      ###
    ###    ._.-.___.' ('\                 ._.-.___.' ('\      ###
    ###   //(   BOB  ( `'                //(  John  ( `'      ###
    ###  '/ )\ ).__. )                  '/ )\ ).__. )         ###
    ###  ' <' `\ ._/'\                  ' <' `\ ._/'\         ###
    ###     `   \     \                    `   \     \        ###
    #############################################################
    ###          __     __                                    ###
    ###         /  \~~~/  \           |\      _,,,---,,_      ###
    ###   ,----(     ..    )    ZZZzz /,`.-'`'Guts-.  ;-;;,_  ###
    ###  /      \__     __/          |,4-  ) )-,_. ,\ (  `'-' ###
    ### /| DAVE    (\  |(           '---''(_/--'  `-'\_)      ###
    ###^ \   /___\  /\ |                                      ###
    ###   |__|   |__|-"                                       ###
    #############################################################                                                   


    print('Here we go !')

    with open(args.config) as fp:
        paths = yaml.full_load(fp)



    params = {'dim': paths['tile_size'],
              'batch_size': paths['batch_size'],
              'n_channels' : paths['n_channels'],
              'n_classes': paths['n_classes'],
              'shuffle': True,
              'paths':paths}

    print('Model generation')

    # Datasets
    path_to_X, path_to_Y = dataset_creation.create_partition(paths['n_img'], paths['n_time'], paths['n_tile'],paths)
    partition =  path_to_X # IDs
    results =  path_to_Y # Labels

    # Generators
    training_generator = dataset_creation.DataGenerator(partition['train'], results, **params)
    validation_generator = dataset_creation.DataGenerator(partition['val'], results, **params)

    model = Sequential()

    
    # Creation of the layers of the CNN

    inputs = tfk.Input(shape=(paths['tile_size'][0], paths['tile_size'][1], 1)) 
    convo = tfk.Conv2D(paths['n_kernels'] , kernel_size=paths['kernel_size'], activation='sigmoid', padding='same', kernel_initializer=paths['kernel_initializer'])(inputs)
    convo = tfk.Conv2D(paths['n_kernels']*2 , kernel_size=paths['kernel_size'], activation='sigmoid', padding='same', kernel_initializer=paths['kernel_initializer'])(convo) 
    convo = tfk.Conv2D(paths['n_kernels']*2*2 , kernel_size=paths['kernel_size'], activation='sigmoid', padding='same', kernel_initializer=paths['kernel_initializer'])(convo)
    convo = tfk.Conv2D(paths['n_kernels']*2 , kernel_size=paths['kernel_size'], activation='sigmoid', padding='same', kernel_initializer=paths['kernel_initializer'])(convo) 
    convo = tfk.Conv2D(paths['n_kernels'] , kernel_size=paths['kernel_size'], activation='sigmoid', padding='same', kernel_initializer=paths['kernel_initializer'])(convo)
    outputs = tfk.Conv2D(1, kernel_size=paths['kernel_size'], padding='same',activation = 'sigmoid')(convo)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    # Creating threshold for metrics
    tr = 0.5

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=paths['learning_rate']),
                      loss=my_loss,#'binary_crossentropy',
                      #loss = ranging_and_tiling_helpers.focal_loss,
                      metrics=[ my_tp,my_fp,my_fn,my_tn,my_precision2,my_recall,tf.keras.metrics.TruePositives(thresholds=tr), 
                               
                               'mae', 
                               'accuracy'])

    # Setup of filepath for logs
    #log_dir = "logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(paths['nExp'])
    log_dir = paths['log_path']+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+args.name
    #os.system('tensorboard --logdir=' + log_dir)
    filepath = log_dir+"/model_"+args.name+".h5"

    # Other settings
    earlystopper = EarlyStopping(monitor="loss",patience=paths['patience'], verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    print('training start')

    if(False):
        X_train=np.load("/home/rfernandez/Bureau/A_Test/DeepLearningRoot/Data_Thibault/data/ML1_input_img00001.time1.number1.npy")
        Y_train=np.load("/home/rfernandez/Bureau/A_Test/DeepLearningRoot/Data_Thibault/data/ML1_result_img00001.time1.number1.npy")
        X = np.empty((10, 512,512), dtype=float)
        Y = np.empty((10,512,512 ), dtype=int)

        for i in range(10):
            X[i,] = X_train
            Y[i,] = Y_train
        
        # Start of the training
        model.fit(X,Y, validation_split=0.33,
                epochs=paths['n_epochs'], 
                callbacks=[tensorboard_callback,earlystopper, checkpoint])


    if(True):
        # Start of the training
        model.fit_generator(training_generator, 
                validation_data=validation_generator, 
                epochs=paths['n_epochs'], 
                callbacks=[tensorboard_callback,earlystopper, checkpoint])

    print('Test done. Model is at : ', filepath)
    print('Now writing result images in logs/',log_dir,'/results/')
    
    model = load_model(filepath, custom_objects={'focal_loss':ranging_and_tiling_helpers.focal_loss})
    #model = load_model(filepath)

    # Each tile will be read and the model will be applied. We then apply the filter bank and save the image
    # This part is flawed since we use all our dataset and not just list_train. Well it'll do for now

    os.mkdir(log_dir+'/results/')

    for img_num in range(1, paths['n_img']+1) :
        if(img_num<10):strI="000"+str(img_num)
        else:
            if(img_num<100):strI="00"+str(img_num)
            else:
                if(img_num<1000):strI="0"+str(img_num)
                else:
                    strI=""+str(img_num)
        list_time = []
        list_tiles = []
        for time_num in range(paths['n_time']) :
            for tile_num in range(paths['n_tile']):
                tile = np.load(paths['dataset_path']+'ML1_input_img0'+strI+'.time'+str(time_num+1)+'.number'+str(tile_num+1)+'.npy')
                list_tiles.append(tile)

            list_tiles = np.array(list_tiles)
            prediction = model.predict(list_tiles)
            image = ranging_and_tiling_helpers.reverse_tiling([1226, 1348], prediction.reshape(paths['n_tile'], 512, 512), 450)    
        
            list_time.append(image)
            list_tiles = []
        list_time = np.array(list_time).reshape(paths['n_time'], 1226, 1348, 1)
        list_time = list_time*2-1
        list_time = ranging_and_tiling_helpers.filter_bank(list_time)

        io.imsave(log_dir+'/results/ML1_Boite_img0'+str(strI)+'.tiff',list_time.astype(np.uint8))#, cmap='gray', vmin=0, vmax=paths['n_time'])
   


if __name__ == "__main__":

    '''    
    - Parameters needed : 
        - Weights needed
        - YML file with all that is currently in paths, and path of this file
        - Optionnal parameters : 
            - Name of experience
            - Number of images loaded
            - Location of logs
            - Location of images

    - Also need exeptions to work with those
    - Need to load the image as a true mask btween 0 and 22 on 32 bit
    '''

    parser = argparse.ArgumentParser(description="DLR_CIRAD Model creator.")
    subparsers = parser.add_subparsers(title='Mode')

    parser_cnnd = subparsers.add_parser('CNN_dataset', help='Compute the CNN')
    parser_cnnd.add_argument("--config", nargs="?", type=str, default="paths.yml", help="Configuration file")
    parser_cnnd.set_defaults(func=CNN_dataset)
    parser_cnnd.add_argument("--name", nargs="?", type=str, default="test", help="Model name (without .h5)")

    args = parser.parse_args()

if hasattr(args, 'func'):
    args.func(args)

# Message d'erreur
    #############################################################
    ###                                                       ###
    ###      __.--**"""**--...__..--**""""*-.                 ### 
    ###    .'                                `-.              ###
    ###  .'                         _           \             ###
    ### /                         .'        .    \   _._      ###
    ###:          Gros Louis      :          :`*.  :-'.' ;    ###
    ###;    `                    ;          `.) \   /.-'      ###
    ###:     `                             ; ' -*   ;         ###
    ###       :.    \           :       :  :        :         ###
    ### ;     ; `.   `.         ;     ` |  '                  ###
    ### |         `.            `. -*"*\; /        :          ###
    ### |    :     /`-.           `.    \/`.'  _    `.        ###
    ### :    ;    :    `*-.__.-*""":`.   \ ;  'o` `. /        ###
    ###       ;   ;                ;  \   ;:       ;:   ,/    ###
    ###  |  | |                       /`  | ,      `*-*'/     ###
    ###  `  : :  :                /  /    | : .    ._.-'      ###
    ###   \  \ ,  \              :   `.   :  \ \   .'         ###
    ###    :  *:   ;             :    |`*-'   `*+-*           ###
    ###    `**-*`""               *---*                       ###
    #############################################################