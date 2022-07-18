#test model
import sys
import os

sys.path.insert(1, os.path.abspath('C:/Users/infer/Documents/CIRAD_stage_2022/Deep_learning_root'))

print(os.getcwd())
#os.chdir('C:/Users/infer/Documents/CIRAD_stage_2022/Deep_learning_root')
#print(os.getcwd())
from PIL import Image
from keras.models import load_model
import paths
import ranging_and_tiling_helpers
import data_prep_3D
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

model = load_model("model/model_test_sequential.h5", custom_objects={'focal_loss':ranging_and_tiling_helpers.focal_loss})

i = ranging_and_tiling_helpers.sanitised_input("Which image do you wanna see ? : ", int, 0, (paths.n_img)*(paths.n_tile)*(paths.n_time))
X_tile, Y_tile=[], []
X_test = np.array(data_prep_3D.create_Xarray(Image.open('input_test.tif')))
for img in X_test : 
    tiles = np.array(ranging_and_tiling_helpers.tiling(img, paths.TILE_SIZE, paths.STRIDE))
    for tile in tiles : 
        X_tile.append(tile)
X_tile = np.array(X_tile)
Y_test = np.array(data_prep_3D.create_Yarray_speedy(Image.open('output_test.tif')))
for img in Y_test : 
    tiles = np.array(ranging_and_tiling_helpers.tiling(img, paths.TILE_SIZE, paths.STRIDE))
    for tile in tiles : 
        Y_tile.append(tile)
Y_tile = np.array(Y_tile)

#model.load_weights(filepath)

print(np.shape(X_tile))
test = model.predict(X_tile[:, :, :])
test_predicted = test[i, :, :, 0]

test_image_source = X_tile[i, :, :]
test_expected = Y_tile[i, :, :]

print('max ', np.max(test_predicted))
print('min ', np.min(test_predicted))
print('hist ', np.histogram(test_predicted))

plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(test_image_source, cmap='gray')
plt.title('Original', fontsize=14)

plt.subplot(1,3,2)
plt.imshow(test_predicted, cmap='gray')
plt.title('Predicted', fontsize=14)

io.imsave('imagetest/img1.tiff', test_predicted.astype(np.float32))


plt.subplot(1,3,3)
plt.imshow(test_expected, cmap='gray')
plt.title('Result we want', fontsize=14)

plt.show()
'''
for time_num in range(paths.n_time) :
        for tile_num in range(paths.n_tile):

prediction = model.predict(X_tile)'''