
import numpy as np
import ranging_and_tiling_helpers
import os

def data_arborescence_setup(list_X, list_Y,paths) : 
    '''
    Create files in the local folder (specified in the YAML file in argument)

    Parameters : 
    ------------

    list_X : 
    list_Y : 
    paths : 
    '''

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