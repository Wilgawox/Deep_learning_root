
import data_prep_3D as dp3
import ranging_and_tiling_helpers as rt
import numpy as np
from PIL import Image

img = Image.open("image_test.tif")

X = dp3.create_Xarray(img)
X_ranged = rt.reduced_centered_range(X[0], -1, 1)
'''k = 4'''# <---- TODO : Put the right value heeere
X_f = rt.tiling(X_ranged, [512, 512], 450)[3]

def test_create_arrayX() :
    assert np.shape(X_f) != [512, 512], 'Problem with  tile generation, probably at the sides'
def test_tiling() : 
    assert np.shape(X)!=[22,np.shape(img)[1], np.shape(img)[0]], 'Problem while transforming TIFF image to array'
def test_range() :
    assert np.max(X_ranged)!=1 or np.min(X_ranged)!=-1, 'Wrong range values' # or X_ranged(x, y) != k


if __name__ == "__main__":
    test_range()
    test_tiling()
    test_create_arrayX()
    print("Everything passed")