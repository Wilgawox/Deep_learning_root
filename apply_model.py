print('START')

#tensorboard --logdir Documents/CIRAD_stage_2022/Deep_learning_root/logs

from xml.parsers.expat import model
import paths # TODO : put this in a separate file with variables
import ranging_and_tiling_helpers

print('imported local files')

import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import argparse

def imsave(model, img) : 
    imsave(img_model_applied_and_filtered)

def imshow(model, img) : 
    imshow(img_model_applied)


'''
model = load_model(filepath, custom_objects={'focal_loss':focal_loss})
#model = load_model(filepath)

# Optionnal visual test for the user to determine if the training is good or not
Inp = input("Do you want to see the model applied to a tile without filter ? (Y/N)")
if Inp == 'Y' or Inp=='y' : 
    i = ranging_and_tiling_helpers.sanitised_input("Which image do you wanna see ? : ", int, 0, (paths.n_img)*(paths.n_tile)*(paths.n_time))
    #model.load_weights(filepath)

    print(np.shape(X_test))
    test = model.predict(X_test[:50, :, :])
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

    del(test, test_image, test_img_pred, test_res_img)

# Each tile will be read and the model will be applied. We then apply the filter bank and save the image
# This part is flawed since we use all our dataset and not just list_train. Well it'll do for now

os.mkdir(log_dir+'/results/')

for img_num in range(paths.n_img) :
    list_time = []
    list_tiles = []
    for time_num in range(paths.n_time) :
        for tile_num in range(paths.n_tile):
            if img_num<9 :
                tile = np.load(paths.dataset_path+'ML1_input_img0'+str(img_num+1)+'.time'+str(time_num)+'.number'+str(tile_num)+'.npy')
            else :
                tile = np.load(paths.dataset_path+'ML1_input_img'+str(img_num+1)+'.time'+str(time_num)+'.number'+str(tile_num)+'.npy')
            list_tiles.append(tile)
        list_tiles = np.array(list_tiles)
        print(np.max(list_tiles[paths.n_tile-1]))
        #plt.imshow(list_tiles[paths.n_tile-1])
        #plt.show()
        prediction = model.predict(list_tiles)
        image = ranging_and_tiling_helpers.reverse_tiling([1226, 1348], prediction.reshape(paths.n_tile, 512, 512), 450)    
        #plt.imshow(image)
        #plt.show()

        list_time.append(image)
        list_tiles = []
    list_time = np.array(list_time).reshape(paths.n_time, 1226, 1348, 1)
    list_time = list_time*2-1
    print(np.max(list_time[21]))
    print(np.min(list_time[21]))
    #plt.imshow(list_time[21])
    #plt.show()
    list_time = ranging_and_tiling_helpers.filter_bank(list_time)
    # See for the images to be save only if the user allows it to
    #plt.imshow(list_time)
    #plt.show()
    plt.imsave(log_dir+'/results/ML1_Boite_'+str(img_num+1)+'.tiff',list_time, cmap='gray')
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLR_CIRAD Image processor")
    subparsers = parser.add_subparsers(title='Functions')

    parser_imsave = subparsers.add_parser('apply_model', help='Save the model applied to an image, after application of a filter bank')
    #parser_imsave.add_argument("--config", nargs="?", type=str, default="paths.py", help="Configuration file")
    parser_imsave.set_defaults(func=imsave)

    parser_imshow = subparsers.add_parser('apply_model', help='Show an image with a model applied')
    #parser_imshow.add_argument("--config", nargs="?", type=str, default="paths.py", help="Configuration file")
    parser_imshow.set_defaults(func=imshow)

    args = parser.parse_args()
    args.func(args)