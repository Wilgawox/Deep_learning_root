
print('START')

#tensorboard --logdir Documents/CIRAD_stage_2022/Deep_learning_root/logs

#python cnn.py CNN

import paths as p # TODO : put this in a separate file with variables
import ranging_and_tiling_helpers
import dataset_config
import dataset_creation
print('imported local files')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import yaml
import tensorflow.keras.layers as tfk
from keras.models import Sequential
from keras.models import load_model
import argparse
from keras.callbacks import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def CNN_dataset(args) : 
    # Create a dataset then use it to train it with a CNN. Then returns a model.h5 in the /logs file along with this model applied to some images.

    print('Here we go !')

    with open(args.config) as fp:
        paths = yaml.full_load(fp)

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
    params = {'dim': (512, 512),
              'batch_size': paths['batch_size'],
              'n_channels' : 1,
              'n_classes': 2,
              'shuffle': False}

    print('Model generation')

    # Datasets
    path_to_X, path_to_Y = dataset_creation.create_partition(paths['n_img'], paths['n_time'], paths['n_tile'])
    partition =  path_to_X # IDs
    results =  path_to_Y # Labels

    # Generators
    training_generator = dataset_creation.DataGenerator(partition['train'], results, **params)
    validation_generator = dataset_creation.DataGenerator(partition['val'], results, **params)

    model = Sequential()

    
    # Creation of the layers of the CNN

    inputs = tfk.Input(shape=(paths['tile_size'][0], paths['tile_size'][1], 1)) 
    convo1 = tfk.Conv2D(paths['n_kernels'] , kernel_size=[3, 3], activation='sigmoid', padding='same', kernel_initializer='he_normal')(inputs)
    convo2 = tfk.Conv2D(paths['n_kernels']*2 , kernel_size=[3, 3], activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo1) 
    convo3 = tfk.Conv2D(paths['n_kernels']*2*2 , kernel_size=[3, 3], activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo2)
    convo4 = tfk.Conv2D(paths['n_kernels']*2 , kernel_size=[3, 3], activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo3) 
    convo5 = tfk.Conv2D(paths['n_kernels'] , kernel_size=[3, 3], activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo4)
    output = tfk.Conv2D(1, 1, activation = 'sigmoid')(convo5)
    model = tf.keras.Model(inputs = inputs, outputs = output)

    #weights = [1, 60]
    print('model created')
    Y_train = Y_train.astype(np.float16)

    # Creating threshold for metrics
    tr = 0.5

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                      loss='binary_crossentropy',
                      #loss = ranging_and_tiling_helpers.focal_loss,
                      metrics=[tf.keras.metrics.Precision(thresholds=tr), tf.keras.metrics.Recall(thresholds=tr), 'mae', 'accuracy'])

    # Setup of filepath for logs
    #log_dir = "logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(paths['nExp'])
    log_dir = "logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+args.exp
    #os.system('tensorboard --logdir=' + log_dir)
    filepath = log_dir+"/model_"+args.exp+".h5"

    # Other settings
    earlystopper = EarlyStopping(patience=paths['patience'], verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorboard_callback,earlystopper, checkpoint]

    print('training start')

    # Start of the training
   model.fit(training_generator, 
                  validation_data=validation_generator, 
                  epochs=paths['n_epochs'],
                  callbacks=[tensorboard_callback,earlystopper, checkpoint])


    print('Test done. Model is at : ', filepath)
    
    model = load_model(filepath, custom_objects={'focal_loss':ranging_and_tiling_helpers.focal_loss})
    #model = load_model(filepath)

    # Each tile will be read and the model will be applied. We then apply the filter bank and save the image
    # This part is flawed since we use all our dataset and not just list_train. Well it'll do for now

    os.mkdir(log_dir+'/results/')

    for img_num in range(1, paths['n_img']+1) :
        list_time = []
        list_tiles = []
        for time_num in range(paths['n_time']) :
            for tile_num in range(paths['n_tile']):
                if(img_num<10):strI="000"+str(img_num)
                else:
                    if(img_num<100):strI="00"+str(img_num)
                    else:
                        if(img_num<1000):strI="0"+str(img_num)
                        else:
                            strI=""+str(img_num)
            tile = np.load(paths['dataset_path']+'ML1_input_img0'+strI+'.time'+str(time_num+1)+'.number'+str(tile_num+1)+'.npy')
            list_tiles.append(tile)

        list_tiles = np.array(list_tiles)
        prediction = model.predict(list_tiles)
        image = ranging_and_tiling_helpers.reverse_tiling([1226, 1348], prediction.reshape(paths['n_tile'], 512, 512), 450)    
        
        list_time.append(image)
        list_tiles = []
    list_time = np.array(list_time).reshape(paths['n_time'], 1226, 1348, 1)
    list_time = list_time*2-1
    print(np.max(list_time[21]))
    print(np.min(list_time[21]))
    #plt.imshow(list_time[21])
    #plt.show()
    list_time = ranging_and_tiling_helpers.filter_bank(list_time)
    # See for the images to be save only if the user allows it to
    #plt.imshow(list_time)
    #plt.show()
    plt.imsave(log_dir+'/results/ML1_Boite_'+str(img_num+1)+'.tiff',list_time.astype(uint8), cmap='gray', vmin=0, vmax=paths['n_time'])


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
    parser_cnnd.set_defaults(func=CNN)
    parser_cnnd.add_argument("--exp", nargs="?", type=str, default="test", help="Model name (without .h5)")

    args = parser.parse_args()
if hasattr(args, 'func'):
    args.func(args)