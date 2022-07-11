# Data paths
training_data = "Data_Thibault/input/"
res_data = "Data_Thibault/results/"
dataset_path = "Data_Thibault/data/"
training_data_path = dataset_path+"data_training/"
val_data_path = dataset_path+"data_validation/"
test_data_path = dataset_path+"data_test/"
MODEL_FILEPATH = "Data_Thibault/model.h5"
final_data = "Data_Thibault/processed_data/"

# Parameters for the tiles
TILE_SIZE = [512,512] #[x,y] dimensions of a tile
IMG_H = TILE_SIZE[0] # TODO : Change this variable name
IMG_W = TILE_SIZE[1] # TODO : Change this variable name
STRIDE =  450 #Stride between two tiles in a image
INT_ROOT = -1 #Target value of a root pixel
INT_BG = 1 #Target value of a backgroung pixel

# Parameters for the CNN :
PERCENT_TRAIN_IMAGES = 50 # Part of image for training (%) Default : 50
PERCENT_VALID_IMAGES = 25 # Part of image for validation (%) Default : 25
PERCENT_TEST_IMAGES = 25 # Part of image for test (%) Default : 25
SAMPLE_WEIGHT = 60 # Weight given for a pixel associated with a root
PATIENCE = 20 # Nomber of unsuccessful epochs before the training stops
nb_epochs = 100 # Total number of epochs
batch_size = 10 # Batch size
validation_split = 0.1
layers = 3 #Number of layers for the CNN
N_CHANNELS = 1
N_CLASSES = 2 #softmax #Root or background
NUM_TEST_IMAGES = 0 #Calcul a faire avant execution

# Test parameters for main :
n_img = 10 # Number of images processed by the training
n_time = 22 # Number of time slices used in each image
n_tile = 9 # Number of tiles processed (TODO : always set it to max)
nExp = "test_focal_loss_100" # Name of the current experiment