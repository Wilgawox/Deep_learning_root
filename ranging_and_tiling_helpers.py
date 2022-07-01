import string
import numpy as np
import paths
import statistics


def reduced_centered_range(img, intensity_bg, intensity_root) :
    #Process the img to adjust the value between the 2 intensities, and reverse the black and white
    maxx = np.amax(img)
    minn = np.amin(img)

    # Apply a ranging operation to each pixel
    a = (intensity_bg-intensity_root)/(maxx-minn)
    b = intensity_bg-a*maxx
    imgCR = a*img+b
    return np.array(imgCR).astype(np.dtype('float32'))


def average_range(img, intensity_bg, intensity_root) :
    # Take an image to fing its max/min values, take the adjacent pixels, and then find the average value of a pixel of root/background
    # Then it change the image to have max(img)=average_root_value and min(img)=average_min_value
    
    # Create maxx value from the first maxima of the image and find its location
    maxx = np.max(img)
    max_loc = np.where(img == maxx)#Location of the max value
    max_loc = list(zip(max_loc[0], max_loc[1]))[0]# Tuple with location of max value
    
    # Create minn value from the first minima of the image and find its location
    minn = np.amin(img)
    min_loc = np.where(img == minn)#Location of the max value
    min_loc = list(zip(min_loc[0], min_loc[1]))[0]# Tuple with location of min value
    
    # Take the 4 nearest pixels of the maximum/minimum, and find the mean value of those
    maxx = statistics.mean([img[max_loc[0]][max_loc[1]+1], img[max_loc[0]][max_loc[1]-1], img[max_loc[0]+1][max_loc[1]], img[max_loc[0]-1][max_loc[1]]])
    minn = statistics.mean([img[min_loc[0]][min_loc[1]+1], img[min_loc[0]][min_loc[1]-1], img[min_loc[0]+1][min_loc[1]], img[min_loc[0]-1][min_loc[1]]])
    
    # Apply a ranging operation to each pixel
    a = (intensity_bg-intensity_root)/(maxx-minn)
    b = intensity_bg-a*maxx
    imgCR = a*img+b
    return np.array(imgCR).astype(np.dtype('float32'))

def tiling(img, final_size : tuple, stride) :
    #Try to slice img to tiles in final_size shape, and slide the rest to be of the same size
    img_w, img_h = img.shape
    tile_w, tile_h = final_size
    tiles=[]
    for i in np.arange(img_w, step=stride):
        for j in np.arange(img_h, step=stride):
            bloc = img[i:i+tile_w, j:j+tile_h]
            if bloc.shape == (tile_w, tile_h):
                # When tile entierely in image
                tiles.append(bloc)
            else :
                # When the tile does not fit into the image
                bloc_w, bloc_h = bloc.shape;
                bloc = img[(i-(tile_w-bloc_w)):(i+tile_w), (j-(tile_h-bloc_h)):(j+tile_h)]
                tiles.append(bloc)
    return tiles

def img_tile_and_range(img : np.array, intensity_bg, intensity_root, tile_size : tuple, stride : int) : 
        img = reduced_centered_range(img, intensity_bg, intensity_root)
        tiles = tiling(img, tile_size, stride)
        return tiles


def reverse_tiling(img_size, tiles, stride) :    
    # This function take an image and its tile list created with the function tiling()
    # We need to keep the same stride used in tiling()
    img_w, img_h = img_size
    tile_w, tile_h = tiles[0].shape
    
    # Creation of 2 variables containing the x/y position of the tiles in the final image
    coordsX=[np.arange(img_w, step=stride)]
    coordsY=[np.arange(img_h, step=stride)]
    
    ntX = len(coordsX[0])
    ntY = len(coordsY[0])
    
    # Adjusting the last coordinate of coordX/coordY
    if(coordsX[0][ntX-1]+tile_w>img_w):
        coordsX[0][ntX-1]=img_w-tile_w
    if(coordsY[0][ntY-1]+tile_h>img_h):
        coordsY[0][ntY-1]=img_h-tile_h
    
    final_img = np.zeros([img_w, img_h])
    
    # Creating a variable to limitate side effects
    semi_overlap_x=(tile_w-stride)//2
    semi_overlap_y=(tile_h-stride)//2

    for i in range(ntX):
        for j in range(ntY):
            cX=coordsX[0][i]
            cY=coordsY[0][j]

            #Identify coordinates into target image
            x0=cX+int(i>0)*semi_overlap_x
            y0=cY+int(j>0)*semi_overlap_y
            xf=cX+tile_w-int(i<(ntX-1))*semi_overlap_x
            yf=cY+tile_h-int(j<(ntY-1))*semi_overlap_y

            #Identify coordinates into source tile
            a0=0+int(i>0)*semi_overlap_x
            b0=0+int(j>0)*semi_overlap_y
            af=tile_w-int(i<(ntX-1))*semi_overlap_x
            bf=tile_h-int(j<(ntY-1))*semi_overlap_y

            tile=tiles[i*ntY+j]
            final_img[x0:xf,y0:yf]=tile[a0:af,b0:bf]
    return final_img


def sanitised_input(prompt, type_=None, min_=None, max_=None, range_=None):
    #Fonction stolen at https://stackoverflow.com/questions/23294658/asking-the-user-for-input-until-they-give-a-valid-response
    # Help getting an int when the user input something in main
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min_ must be less than or equal to max_.")
    while True:
        ui = input(prompt)
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) == 1:
                    print(template.format(*range_))
                else:
                    expected = " or ".join((
                        ", ".join(str(x) for x in range_[:-1]),
                        str(range_[-1])
                    ))
                    print(template.format(expected))
        else:
            return ui


def filter_bank(time_sequence):
    # Input  : a np.array( dim_X, dim_Y, dim_T) with float values indicating probability of background (values near -1) or root (values near 1)
    # Output : a np.array( dim_X, dim_Y ) with integer values indicating for each pixel (x,y) the root apparition time from 1 to max_time, or zero if no_root
    
    N_times=np.shape(time_sequence)[0]

    # The filter_bank is a list of signal models corresponding to apparition of a root, computed for each target time
    filter_bank=np.array([[[[ (j*2-1) if(j<2) else (-1+2*int( i>(j-2)))  for j in range(N_times+1)] ] ] for i in range(N_times)] )  # F Y X T

    # The broadcasted element-wise dotproduct sum(data-mult-bank filter) try all the filters of the bank for each pixel to estimate the likelihood 
    # of a root apparition at each target time. Then we use argmax function to select the index of the filter which gave the highest response
    return np.argmax(np.sum( np.multiply(time_sequence,filter_bank),axis=0),axis=2)


## Alternative fonction to tiling, to complete if needed

#def tiling_only_roots(img : np.array, final_size : tuple, stride) :
#    #Slice img to tiles in final_size shape, and keep only the slices that have white pixel in it, not using most of the tiles without root in them
#    # NOT YET TESTED
#    img_w, img_h = img.shape
#    tile_w, tile_h = final_size
#    tiles=[]
#    for i in np.arange(img_w, step=stride):
#        for j in np.arange(img_h, step=stride):
#            bloc = img[i:i+tile_w, j:j+tile_h]
#            if np.max(bloc)>(paths.INT_ROOT+paths.INT_BG)/2 :
#                if bloc.shape == (tile_w, tile_h):
#                    tiles.append(bloc)
#                else :
#                    bloc_w, bloc_h = bloc.shape;
#                    bloc = img[(i-(tile_w-bloc_w)):(i+tile_w), (j-(tile_h-bloc_h)):(j+tile_h)]
#                    tiles.append(bloc)
#    return tiles
