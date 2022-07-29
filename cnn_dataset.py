
print('START')

# Dataset creation based on : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

#python cnn_dataset.py CNN_dataset --name name

import tensorflow as tf

import ranging_and_tiling_helpers
import custom_metrics_and_losses
import dataset_creation
import model_utils

import os
import numpy as np
from skimage import io
import datetime
import yaml
import tensorflow.keras.layers as tfk
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


print("Imports done")

def CNN_dataset(args) : 
    '''
    CNN_dataset is used when you already created images files in your local repository, and need to train your dataset.
    '''

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

    # Datasets
    path_to_X, path_to_Y = dataset_creation.create_partition(paths['n_img'], paths['n_time'], paths['n_tile'],paths)
    partition =  path_to_X # IDs
    results =  path_to_Y # Labels

    # Generators
    training_generator = dataset_creation.DataGenerator(partition['train'], results, **params)
    validation_generator = dataset_creation.DataGenerator(partition['val'], results, **params)



    # Setup of filepath for logs
    log_dir = paths['log_path']+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_'+args.name
    filepath = log_dir+args.name+".h5"

    # Other settings
    earlystopper = EarlyStopping(monitor="loss",patience=paths['patience'], verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    

    inputs, outputs = model_utils.resnet(paths,paths['depth_resnet'])
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=paths['learning_rate']),
                    # loss ='binary_crossentropy',
                    loss = ranging_and_tiling_helpers.f1_loss,
                    metrics=[ custom_metrics_and_losses.recall_custom,
                            custom_metrics_and_losses.precision_custom,
                            'accuracy'])


    # Start of the training
    model.fit_generator(training_generator, 
            validation_data=validation_generator, 
            epochs=paths['n_epochs'], 
            callbacks=[tensorboard_callback,earlystopper, checkpoint])

    print('Test done. Model is at : ', filepath)
    print('Now writing result images in logs/',log_dir,'/results/')
    
    model = load_model(filepath, custom_objects=paths['custom_obj'])


    # Each tile will be read and the model will be applied on the iimages the training did not learn on. We then apply the filter bank and save the image
    os.mkdir(log_dir+'/results/')

    for img_num in range(1, (paths['n_img']/2)+1) :
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
                tile = np.load(paths['dataset_path']+'ML1_input_img0'+strI+'_time'+str(time_num+1)+'_number'+str(tile_num+1)+'.npy')
                list_tiles.append(tile)

            list_tiles = np.array(list_tiles)
            prediction = model.predict(list_tiles)
            image = ranging_and_tiling_helpers.reverse_tiling([1226, 1348], prediction.reshape(paths['n_tile'], 512, 512), 450)    
        
            list_time.append(image)
            list_tiles = []
        list_time = np.array(list_time).reshape(paths['n_time'], 1226, 1348, 1)
        list_time = list_time*2-1
        list_time = ranging_and_tiling_helpers.filter_bank(list_time)

        io.imsave(log_dir+'/results/ML1_Boite_img0'+str(strI)+'.tiff',list_time.astype(np.uint8))
   


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
    '''

    parser = argparse.ArgumentParser(description="DLR_CIRAD Model creator.")
    subparsers = parser.add_subparsers(title='Mode')

    parser_cnnd = subparsers.add_parser('CNN_dataset', help='Compute the CNN')
    parser_cnnd.add_argument("--config", nargs="?", type=str, default="paths.yml", help="Configuration file")
    parser_cnnd.set_defaults(func=CNN_dataset) 
    parser_cnnd.add_argument("--dev", nargs="?", type=bool, default=False, help="Developper mode")
    parser_cnnd.add_argument("--name", nargs="?", type=str, default="test", help="Model name (without .h5)")

    args = parser.parse_args()

if hasattr(args, 'func'):
    args.func(args)

# Message d'erreur
    ##############################################################
    ###                                                        ###
    ###       __.--**"""**--...__..--**""""*-.                 ### 
    ###     .'                                `-.              ###
    ###   .'                         _           \             ###
    ###  /                         .'        .    \   _._      ###
    ### :          Big Louis      :          :`*.  :-'.' ;     ###
    ### ;    `                    ;          `.) \   /.-'      ###
    ### :     `                             ; ' -*   ;         ###
    ###        :.    \           :       :  :        :         ###
    ###  ;     ; `.   `.         ;     ` |  '                  ###
    ###  |         `.            `. -*"*\; /        :          ###
    ###  |    :     /`-.           `.    \/`.'  _    `.        ###
    ###  :    ;    :    `*-.__.-*""":`.   \ ;  'o` `. /        ###
    ###        ;   ;                ;  \   ;:       ;:   ,/    ###
    ###   |    |                       /`  | ,      `*-*'/     ###
    ###   `  : :  :                /  /    | : .    ._.-'      ###
    ###    \  \ ,  \              :   `.   :  \ \   .'         ###
    ###     :  *:   ;             :    |`*-'   `*+-*           ###
    ###     `**-*`""               *---*                       ###
    ##############################################################