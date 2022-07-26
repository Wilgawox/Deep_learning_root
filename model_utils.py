import tensorflow.keras.layers as tfk

# Def losses
# Creation of layers using Resnet
def resnet(paths,depth_resnet) :
    input_res = tfk.Input(shape=(paths['tile_size'][0], paths['tile_size'][1], 1)) 
    inputs=input_res

    for i in range(depth_resnet):
        inputs_shortcut=inputs
        convo = tfk.Conv2D(paths['n_kernels'] , kernel_size=paths['kernel_size'], padding='same', kernel_initializer=paths['kernel_initializer'])(inputs)
        convo=tfk.BatchNormalization(axis=3)(convo)
        convo=tfk.Activation(activation=paths['activation'])(convo)
        convo = tfk.Conv2D(paths['n_kernels'] , kernel_size=paths['kernel_size'], padding='same', kernel_initializer=paths['kernel_initializer'])(convo)
        convo=tfk.BatchNormalization(axis=3)(convo)
        inputs=tfk.Add()([inputs_shortcut,convo])

    outputs = tfk.Conv2D(1, kernel_size=1, padding='same',activation = 'sigmoid')(convo)
    return(input_res, outputs)


