print('START')

#python make_data.py create_files

from xml.parsers.expat import model
import paths # TODO : put this in a separate file with variables
import data_prep_3D
import data_prep_2D

from PIL import Image
import argparse

print('Imports finished')


def create_files(args) :

    list_X = []
    list_Y = []

    for i in range(1,paths.n_img+1) :

        if(i<10):strI="000"+str(i)
        else:
            if(i<100):strI="00"+str(i)
            else:
                if(i<1000):strI="0"+str(i)
                else:
                    strI=""+str(i)
        
        filenameX=paths.training_data+'ML1_Boite_0'+strI+'.tif'
        imX=Image.open(filenameX)
        list_X.append(data_prep_3D.create_Xarray(imX))

        filenameY=paths.res_data+'ML1_Boite_0'+strI+'.tif'
        imY=Image.open(filenameY)
        list_Y.append(data_prep_3D.create_Yarray_speedy(imY))

    #print(list_X, list_Y)
    data_prep_2D.data_arborescence_setup(list_X, list_Y)
    # Free memory
    del(list_X)
    del(list_Y)

    print('Data prep done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data Creator',description="DLR_CIRAD Data generator")
    subparsers = parser.add_subparsers(title='Action')

    parser_cnn = subparsers.add_parser('create_files', help='Create files for X and Y in the designed folder (Folder is WIP)')
    #parser_cnn.add_argument("--config", nargs="?", type=str, default="paths.py", help="Configuration file")
    parser_cnn.set_defaults(func=create_files)

    args = parser.parse_args()
    print(args)
    args.func(args)