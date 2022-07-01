print('START')

from xml.parsers.expat import model
import paths # TODO : put this in a separate file with variables
import glob
import data_prep_3D
import ranging_and_tiling_helpers
import dataset_config
import data_prep_2D

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime
import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as tfk 
from keras.callbacks import *


'''
TODO : Create a test for this

#I1 = np.load("Data_Thibault\data\ML1_input_img0.time0.number0.npy")
#I2 = Image.open("Data_Thibault\Results\ML1_Boite_00009.tif")
#A = plt.plot([0,1],[0,1])
#plt.imshow(I1)
#plt.show()


#X = data_prep_3D.create_Xarray(I1)
#Y = data_prep_3D.create_Yarray(I2)
##plt.imshow(X[20])
##plt.show()
##plt.imshow(Y[20])
#X = modif_image.raange(X[3],1,-1)
#'''


Inp = input("Do you want to create .npy files ? (Y/N)")

if Inp == 'Y' or Inp=='y' : 
    print('Writing X_array')
    for i in range(5) :
        list_X = []
        for filename in glob.glob(paths.training_data+'ML1_Boite_000'+i+'*.tif'):
            print('a')
            im=Image.open(filename)
            list_X.append(data_prep_3D.create_Xarray(im))
    
        print('Wrinting Y_array')
        list_Y = []
        for filename in glob.glob(paths.res_data+'*.tif'):
            im=Image.open(filename)
            list_Y.append(data_prep_3D.create_Yarray_speedy(im))
    
        print('Writing all the data. Please check memory to allow for ~12Go data')
        data_prep_2D.data_arborescence_setup(list_X, list_Y)
        # Free memory
        del(list_X)
        del(list_Y)
        print(globals())
        print(locals())

print('Data prep done (or skipped)')

# Getting the data ready to be used by the dataset (shuffled and ordered)
X,Y = dataset_config.create_IO_for_CNN_training(paths.n_img, paths.n_time, paths.n_tile)
X_train,Y_train,X_test,Y_test,X_valid,Y_valid=dataset_config.shuffle_XY(X, Y, paths.PERCENT_TRAIN_IMAGES, paths.PERCENT_VALID_IMAGES, paths.PERCENT_TEST_IMAGES)

# Creation of the layers of the CNN
inputs = tfk.Input(shape=(paths.IMG_W, paths.IMG_H, 1))
convo1 = tfk.Conv2D(paths.batch_size, paths.layers, activation='sigmoid', padding='same', kernel_initializer='he_normal')(inputs)
convo2 = tfk.Conv2D(paths.batch_size, paths.layers, activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo1)
output = tfk.Conv2D(1, 1, activation = 'sigmoid')(convo2)
model = tf.keras.Model(inputs = inputs, outputs = output)

print('model created')

# Setup of metrics, loss, and optimizer
model.compile(optimizer='rmsprop',
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])

# Setup of filepath for logs
log_dir = "logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(paths.nExp)
#os.system('tensorboard --logdir=' + log_dir)
filepath = log_dir+"/model_"+str(paths.nExp)+".h5"

# Other settings
earlystopper = EarlyStopping(patience=paths.PATIENCE, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks_list = [tensorboard_callback,earlystopper, checkpoint]

print('training start')

# Start of the training
history = model.fit(np.array(X_train), 
                   np.array(Y_train), 
                   validation_split=paths.validation_split, batch_size=paths.batch_size, 
                   epochs=paths.nb_epochs, callbacks=callbacks_list)


'''
# Optionnal visual test for the user to determine if the training is good or not
Inp = input("Do you want to see the model applied to a tile without filter ? (Y/N)")
if Inp == 'Y' or Inp=='y' : 
    i = ranging_and_tiling_helpers.sanitised_input("Which image do you wanna see ? : ", int, 0, (paths.n_img)*(paths.n_tile)*(paths.n_time))
    model.load_weights(filepath)
    test = model.predict(X_test)
    test_img_pred = test[i, :, :, 0]
    test_image = X_test[i, :, :]
    test_res_img = Y_test[i, :, :]
    plt.figure(figsize=(10,10))

    plt.subplot(1,3,1)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original', fontsize=14)
    plt.subplot(1,3,2)

    plt.imshow(test_img_pred, cmap='gray')
    plt.title('Predicted', fontsize=14)
    
    plt.subplot(1,3,3)
    plt.imshow(test_res_img, cmap='gray')
    plt.title('Result we want', fontsize=14)

    plt.show()
'''
# Each tile will be read and the model will be applied. We then apply the filter bank and save the image
# This part is flawed since we use all our dataset and not just list_train. Well it'll do for now

os.mkdir(log_dir+'/results/')

for img_num in range(paths.n_img) :
    list_time = []
    list_tiles = []
    for time_num in range(paths.n_time) :
        for tile_num in range(paths.n_tile):
            tile = np.load(paths.dataset_path+'ML1_input_img'+str(img_num)+'.time'+str(time_num)+'.number'+str(tile_num)+'.npy')
            list_tiles.append(tile)
        list_tiles = np.array(list_tiles)
        prediction = model.predict(list_tiles)
        image = ranging_and_tiling_helpers.reverse_tiling([1226, 1348], prediction.reshape(9, 512, 512), 450)    
        list_time.append(image)
        list_tiles = []
    list_time = np.array(list_time).reshape(22, 1226, 1348, 1)
    list_time = list_time*2-1
    list_time = ranging_and_tiling_helpers.filter_bank(list_time)
    # See for the images to be save only if the user allows it to
    plt.imsave(log_dir+'/results/ML1_Boite_'+str(img_num+1)+'.tiff',list_time, cmap='gray')
