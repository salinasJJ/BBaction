import tensorflow as tf
from tensorflow.keras import layers, regularizers

from utils import get_frozen_params, get_params


PARAMS = get_params('Models')
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Models',
        version=PARAMS['version'],
    )


def regularizer():
    if PARAMS['regularization'] == 'weight_decay':
        return None
    elif PARAMS['regularization'] == 'l2':
        return regularizers.l2(PARAMS['weight_decay'])

def get_model(loaded):
    inputs = loaded.layers[0].output
    x = layers.Conv3D(
        filters=PARAMS['num_labels'], 
        kernel_size=(1,1,1), 
        strides=(1,1,1), 
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(),
        data_format=PARAMS['data_format'],
        name='conv3d_test',
        )(loaded.layers[-4].output)
    x = layers.Flatten(PARAMS['data_format'])(x)
    outputs = layers.Activation(
        'linear', 
        dtype='float32',
        )(x)
    return tf.keras.Model(inputs, outputs)
