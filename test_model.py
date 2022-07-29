import os
import glob
import ranging_and_tiling_helpers
import numpy as np
from skimage import io
from tensorflow.keras.models import load_model
import argparse

def test_model(log_dir, paths) :     
    model = load_model(log_dir+'*.h5', custom_objects=paths['custom_obj'])
    
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
    parser = argparse.ArgumentParser('Model_Test')
    subparsers = parser.add_subparsers(title='Parameters')

    parser_cnn = subparsers.add_parser('test_model', help='Take a model and save images with filter and model applied to it in the logs/ folder')
    parser_cnn.add_argument("--config", nargs="?", type=str, default="paths.yml", help="Configuration file")

    last_modified_logs = max(glob.glob('logs/*'), key=os.path.getctime)

    parser_cnn.add_argument("--logs", nargs="?", type=str, default=last_modified_logs+"/", help="Path to where the log containing the model.h5")
    parser_cnn.set_defaults(func=test_model)

    args = parser.parse_args()
    print(args)
    args.func(args.logs, args.config)