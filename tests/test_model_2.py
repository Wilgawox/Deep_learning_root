#test model
import sys
import os

sys.path.insert(1, os.path.abspath('C:/Users/infer/Documents/CIRAD_stage_2022/Deep_learning_root'))

print(os.getcwd())
#os.chdir('C:/Users/infer/Documents/CIRAD_stage_2022/Deep_learning_root')
#print(os.getcwd())
from PIL import Image
from keras.models import load_model
import ranging_and_tiling_helpers
import data_prep_3D
import custom_metrics_and_losses
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

model_name = 'resnet_fast_learning.h5'
stride = 450

custom_obj = {'focal_loss':custom_metrics_and_losses.focal_loss,
             'precision_custom':custom_metrics_and_losses.precision_custom,
             'recall_custom':custom_metrics_and_losses.recall_custom, 
             'bce':custom_metrics_and_losses.bce_custom, 
             'f1':custom_metrics_and_losses.f1, 
             'f1_loss':custom_metrics_and_losses.f1_loss, 
             'weighted_categorical_crossentropy':custom_metrics_and_losses.weighted_categorical_crossentropy}

model = load_model("tests/model/"+model_name, custom_objects=custom_obj)

#i = ranging_and_tiling_helpers.sanitised_input("Which image do you wanna see ? : ", int, 0, (paths.n_img)*(paths.n_tile)*(paths.n_time))
i=180
X_tile, Y_tile=[], []

X_test = np.array(data_prep_3D.create_Xarray(Image.open('tests/input_test.tif')))

for img in X_test : 
    tiles = np.array(ranging_and_tiling_helpers.img_tile_and_range(img, -1, 1, [512, 512], stride))
    for tile in tiles : 
        X_tile.append(tile)
X_tile = np.array(X_tile)

Y_test = np.array(data_prep_3D.create_Yarray_speedy(Image.open('tests/output_test.tif')))

for img in Y_test : 
    tiles = np.array(ranging_and_tiling_helpers.tiling(img, [512, 512], stride))
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
plt.subplot(1,4,1)
plt.imshow(test_image_source, cmap='gray', vmin=-1, vmax=1)
plt.title('Original', fontsize=14)

plt.subplot(1,4,2)
plt.imshow(test_predicted, cmap='gray', vmin=0, vmax=1)
plt.title('Prediction', fontsize=14)

io.imsave('tests/imagetest/img_pred.tiff', test_predicted.astype(np.float32))

plt.subplot(1,4,3)
plt.imshow(test_predicted, cmap='gray', vmin=0.49999, vmax=0.500001)
plt.title('Predicted segmentation', fontsize=14)

#io.imsave('tests/imagetest/img_seg.tiff', test_predicted.astype(np.float32),  vmin=0.49999, vmax=0.500001)


plt.subplot(1,4,4)
plt.imshow(test_expected, cmap='gray', vmin=0, vmax=1)
plt.title('Result we want', fontsize=14)

plt.show()