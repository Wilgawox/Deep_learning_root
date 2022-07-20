from PIL import Image
from skimage import io
import numpy as np

img_8b = Image.open('tests/imagetest/test_save_load/input/img_8b.tif')
img_32bf = Image.open('tests/imagetest/test_save_load/input/img_32bf.tif')

np_8b = np.array(img_8b)
np_32bf = np.array(img_32bf)

io.imsave('tests/imagetest/test_save_load/output/img_8b.tiff', np_8b.astype(np.uint))
io.imsave('tests/imagetest/test_save_load/output/img_32bf.tiff', np_32bf.astype(np.float32))
