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


def create_partition(n_img, time, tile_number) :
    # Former dataset creator, now deprecated
    X_train = []
    X_val =  []
    X_test = []
    Y = {}
        
    for i in range(1, n_img+1) : 
        for t in range(time) :
            for n in range(tile_number) :
                if(i<10) : strI="000"+str(i)
                else :
                    if(i<100) : strI="00"+str(i)
                    else :
                        if(i<1000) : strI="0"+str(i)
                        else : strI=""+str(i)
                        
                a = paths.dataset_path+'ML1_input_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                b = paths.dataset_path+'ML1_result_img0'+strI+'.time'+str(t+1)+'.number'+str(n+1)+'.npy'
                                
                if (i-1)*tile_number*time+tile_number*t+n>=(n_img*time*tile_number)/2 : X_train.append(a) # Put vals from paths for percentage
                elif (i-1)*tile_number*time+tile_number*t+n<=(n_img*time*tile_number)/4 : X_val.append(a) # Put vals from paths for percentage
                else : X_test.append(a)
                Y.update({a : b})
    X = {'train':tuple(X_train), 'val':tuple(X_val), 'test':tuple(X_test)}
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

"""
def CNN(n_img, time, tile_number) :
    # TODO : Split this function
    # train_CNN and apply_CNN
    X,Y = create_IO_for_CNN_training(n_img, time, tile_number)
    X_train,Y_train,X_test,Y_test,X_valid,Y_valid=shuffle_XY(X, Y, paths.PERCENT_TRAIN_IMAGES, paths.PERCENT_VALID_IMAGES, paths.PERCENT_TEST_IMAGES)
    inputs = tfk.Input(shape=(512, 512, 1))
    convo1 = tfk.Conv2D(paths.batch_size, paths.layers, activation='sigmoid', padding='same', kernel_initializer='he_normal')(inputs)
    #convo1 = tfk.Dropout(0.1)(convo1)
    convo2 = tfk.Conv2D(paths.batch_size, paths.layers, activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo1)

    output = tfk.Conv2D(1, 1, activation = 'sigmoid')(convo2)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    model.compile(optimizer='rmsprop',
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    filepath = paths.MODEL_FILEPATH

    # Calculate the weights for each class so that we can balance the data
    #print(np.shape(sample_weight))
    #print(np.shape(X_train), np.shape(Y_train))
    
    # Add the class weights to the training                                         
    sample_weight = np.ones(np.shape(Y_train))
    sample_weight[Y_train == 1] = paths.SAMPLE_WEIGHT

    
    earlystopper = EarlyStopping(patience=paths.PATIENCE, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorboard_callback,earlystopper, checkpoint]

    history = model.fit(np.array(X_train), 
                        np.array(Y_train), 
                        validation_split=paths.validation_split, batch_size=paths.batch_size, 
                        epochs=paths.nb_epochs, callbacks=callbacks_list)

    model.load_weights(paths.MODEL_FILEPATH)
    test = model.predict(X_test)
    test_img_pred = test[0, :, :, 0]
    test_image = X_test[0, :, :]
    test_res_img = Y_test[0, :, :]
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
"""