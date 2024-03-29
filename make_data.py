print('START')

#python make_data.py create_files

#import paths
import data_prep_3D
import data_prep_2D

from PIL import Image
import argparse
import yaml

print('Imports finished')


def create_files(args) :
    '''
    Create a large number of files, each one being one tile of one slice of one .TIFF root photograph.
    Also take the annoted masks and make tiles out of them
    '''
    with open(args.config) as fp:
        paths = yaml.full_load(fp)

    list_X = []
    list_Y = []

    for i in range(1,paths['n_img']+1) :

        # Creation of filename (e.g. : 'ML1_Boite_00015.tif')
        if(i<10):strI="000"+str(i)
        else:
            if(i<100):strI="00"+str(i)
            else:
                if(i<1000):strI="0"+str(i)
                else:
                    strI=""+str(i)
        filenameX=paths['training_data']+'ML1_Boite_0'+strI+'.tif'
        
        # Import the image
        imX=Image.open(filenameX)
        
        # Add the right file to list_X
        list_X.append(data_prep_3D.create_Xarray(imX))

        # Same process for list_Y
        filenameY=paths['res_data']+'ML1_Boite_0'+strI+'.tif'
        imY=Image.open(filenameY)
        list_Y.append(data_prep_3D.create_Yarray_speedy(imY))

    data_prep_2D.data_arborescence_setup(list_X, list_Y,paths)

    # Free memory
    del(list_X)
    del(list_Y)

    print('Data prep done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data Creator',description="DLR_CIRAD Data generator")
    subparsers = parser.add_subparsers(title='Action')

    parser_cnn = subparsers.add_parser('create_files', help='Create files for X and Y in the designed folder (Folder is WIP)')
    parser_cnn.add_argument("--config", nargs="?", type=str, default="paths.yml", help="Configuration file")
    parser_cnn.set_defaults(func=create_files)

    args = parser.parse_args()
    print(args)
    args.func(args)