
import numpy as np
import ranging_and_tiling_helpers
import os
#import paths


def data_arborescence_setup_splitter(list_X, list_Y,paths) :
    #Function not working anymore, keeping it here for now if I need it later on
    # Process and save the images, putting them in different folders to serve differents puposes
   for i in range(0, len(list_X)) :
        for j in range(len(list_X[i])):
            tilesX = ranging_and_tiling_helpers.data_range_and_tile(list_X[i][j], paths.INT_ROOT, paths.INT_BG, paths.TILE_SIZE, paths.STRIDE)
            tilesY = ranging_and_tiling_helpers.tiling(list_Y[i][j], paths.TILE_SIZE, paths.STRIDE)
            if len(tilesX)!=len(tilesY) :
                # Check for different size tiles, which would be a big problem later on
                raise Exception('Problem while cutting tiles', 'Different number of tiles')
            for k in range(len(tilesX)):
                if i<=len(list_X)//2 :
                    #We put 50% of images in training ...
                    np.save((paths.training_data_path+'input/ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                    np.save((paths.training_data_path+'results/ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])
                elif i>3*len(list_X)//4 :
                    #... then 25% of images in test ...
                    np.save((paths.test_data_path+'input/ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                    np.save((paths.test_data_path+'results/ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])
                else :
                    #... and the last 25% in validation
                    np.save((paths.val_data_path+'input/ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                    np.save((paths.val_data_path+'results/ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])



def data_arborescence_setup(list_X, list_Y,paths) : 
    # Save X and Y as .npy in the dataset path, tiled and ranged
    if(not os.path.exists(paths['dataset_path'])) : os.mkdir(paths['dataset_path'])
    for i in range(1, len(list_X)+1) :
        print(i)
        for j in range(1, len(list_X[i-1])+1):
            # Ranging and tiling the images
            tilesX = ranging_and_tiling_helpers.img_tile_and_range(list_X[i-1][j-1], paths['int_root'], paths['int_bg'], paths['tile_size'], paths['stride'])
            tilesY = ranging_and_tiling_helpers.tiling(list_Y[i-1][j-1], paths['tile_size'], paths['stride'])
            if len(tilesX)!=len(tilesY) :
                # Check for different size tiles, which would be a big problem later on
                raise Exception('Problem while cutting tiles', 'Different number of tiles')
            for k in range(1, len(tilesX)+1):
                if(i<10):strI="000"+str(i)
                else:
                    if(i<100):strI="00"+str(i)
                    else:
                        if(i<1000):strI="0"+str(i)
                        else:
                            strI=""+str(i)
                print((paths['dataset_path']+'ML1_input_img0'+strI+'.time'+str(j)+'.number'+str(k)))
                np.save((paths['dataset_path']+'ML1_input_img0'+strI+'.time'+str(j)+'.number'+str(k)), tilesX[k-1])
                np.save((paths['dataset_path']+'ML1_result_img0'+strI+'.time'+str(j)+'.number'+str(k)), tilesY[k-1])