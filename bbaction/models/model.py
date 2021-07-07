import tensorflow as tf
from tensorflow.keras import layers, regularizers

from utils import get_frozen_params, get_params


PARAMS = get_params('Models')
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Models',
        version=PARAMS['version'],
    )

FILTER_DIM = 1 if PARAMS['data_format'] == 'channels_first' else -1
NUM_BLOCKS = {
    26: (2, 2, 2, 2),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}
FILTERS = [256, 512, 1024, 2048]
STRIDES = [
    (1, 1, 1),
    (2, 2, 2),
    (1, 2, 2),
]


def regularizer():
    if PARAMS['regularization'] == 'weight_decay':
        return None
    elif PARAMS['regularization'] == 'l2':
        return regularizers.l2(PARAMS['weight_decay'])

def block(inputs, stride_idx):
    if PARAMS['arch'] in ['ir-csn', 'ir']:
        x = layers.ZeroPadding3D(
            padding=[1,1,1],
            data_format=PARAMS['data_format'],
            )(inputs)
        x = layers.Conv3D(
            filters=x.shape[FILTER_DIM],
            kernel_size=[3,3,3],
            strides=STRIDES[stride_idx],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(),
            data_format=PARAMS['data_format'],
            groups=x.shape[FILTER_DIM],
            )(x)
    elif PARAMS['arch'] in ['ip-csn', 'ip']:
        x = layers.Conv3D(
            filters=inputs.shape[FILTER_DIM],
            kernel_size=[1,1,1],
            strides=[1,1,1],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(),
            data_format=PARAMS['data_format'],
            )(inputs)
        x = layers.BatchNormalization(
            momentum=PARAMS['bn_momentum'],
            epsilon=PARAMS['epsilon'],
            scale=False,
            gamma_regularizer=regularizer(),
            beta_regularizer=regularizer(),
            )(x)
        x = layers.ReLU()(x)

        x = layers.ZeroPadding3D(
            padding=[1,1,1],
            data_format=PARAMS['data_format'],
            )(x)
        x = layers.Conv3D(
            filters=x.shape[FILTER_DIM],
            kernel_size=[3,3,3],
            strides=STRIDES[stride_idx],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(),
            data_format=PARAMS['data_format'],
            groups=x.shape[FILTER_DIM],
            )(x)
    return x
        
def bottleneck(
        inputs, 
        filter_idx,
        stride_idx
    ):
    num_filters = FILTERS[filter_idx] \
                        if PARAMS['depth'] >= 50 else FILTERS[filter_idx] / 4
    skip = inputs

    x = layers.Conv3D(
        filters=int(num_filters / 4) if PARAMS['depth'] >= 50 else num_filters,
        kernel_size=[1,1,1],
        strides=[1,1,1],
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(),
        data_format=PARAMS['data_format'],
        )(inputs)
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(),
        beta_regularizer=regularizer(),
        )(x)
    x = layers.ReLU()(x)

    x = block(
        x, 
        stride_idx=stride_idx,
    )
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(),
        beta_regularizer=regularizer(),
        )(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(
        filters=num_filters,
        kernel_size=[1,1,1],
        strides=[1,1,1],
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(),
        data_format=PARAMS['data_format'],
        )(x)
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(),
        beta_regularizer=regularizer(),
        )(x)

    if skip.shape[FILTER_DIM] != num_filters:
        skip = layers.Conv3D(
            filters=num_filters,
            kernel_size=[1,1,1],
            strides=STRIDES[stride_idx],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(),
            data_format=PARAMS['data_format'],
            )(skip)
        skip = layers.BatchNormalization(
            momentum=PARAMS['bn_momentum'],
            epsilon=PARAMS['epsilon'],
            scale=False,
            gamma_regularizer=regularizer(),
            beta_regularizer=regularizer(),
            )(skip)
    x = layers.Add()([
        skip, 
        x
    ])
    x = layers.ReLU()(x)
    return x
    
def network(inputs):
    x = layers.ZeroPadding3D(
        padding=(1,3,3),
        data_format=PARAMS['data_format'],
        )(inputs)
    x = layers.Conv3D(
        filters=64, 
        kernel_size=(3,7,7), 
        strides=(1,2,2), 
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(),
        data_format=PARAMS['data_format'],
        )(x)
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(),
        beta_regularizer=regularizer(),
        )(x)
    x = layers.ReLU()(x)
    
    x = layers.ZeroPadding3D(
        padding=(0,1,1),
        data_format=PARAMS['data_format'],
        )(x)
    x = layers.MaxPool3D(
        pool_size=(1,3,3),
        strides=(1,2,2),
        data_format=PARAMS['data_format'],
        )(x)
        
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][0]):
        x = bottleneck(
            x, 
            filter_idx=tf.constant(0),
            stride_idx=tf.constant(0),
        )
    
    x = bottleneck(
        x, 
        filter_idx=tf.constant(1), 
        stride_idx=tf.constant(1),
    )
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][1] - 1):
        x = bottleneck(
            x, 
            filter_idx=tf.constant(1),
            stride_idx=tf.constant(0),
        )
    
    if PARAMS['frames_per_clip'] < 4:
        x = bottleneck(
            x, 
            filter_idx=tf.constant(2), 
            stride_idx=tf.constant(2),
        )
    else:
        x = bottleneck(
            x, 
            filter_idx=tf.constant(2),
            stride_idx=tf.constant(1),
        )
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][2] - 1):
        x = bottleneck(
            x, 
            filter_idx=tf.constant(2),
            stride_idx=tf.constant(0),
        )
   
    if PARAMS['frames_per_clip'] < 8:
        x = bottleneck(
            x, 
            filter_idx=tf.constant(3),
            stride_idx=tf.constant(2),
        )
    else:
        x = bottleneck(
            x, 
            filter_idx=tf.constant(3), 
            stride_idx=tf.constant(1),
        )
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][3] - 1):
        x = bottleneck(
            x, 
            filter_idx=tf.constant(3),
            stride_idx=tf.constant(0),
        )

    x = layers.AveragePooling3D(
        pool_size=[
            PARAMS['frames_per_clip'] // 8 \
                if PARAMS['frames_per_clip'] >= 8 else 1, 
            7, 
            7,
        ], 
        strides=[1, 1, 1],
        data_format=PARAMS['data_format'],
        )(x)
    x = layers.Dropout(PARAMS['dropout_rate'])(x)
    x = layers.Flatten(data_format=PARAMS['data_format'])(x)
    x = layers.Dense(
            PARAMS['num_labels'],
            use_bias=PARAMS['use_bias'], 
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(),
            )(x)
    x = layers.Activation(
        'linear', 
        dtype='float32',
        )(x)
    return x

def get_model():
    if PARAMS['data_format'] == 'channels_first':
        inputs = tf.keras.Input([
            3,
            PARAMS['frames_per_clip'], 
            PARAMS['spatial_size'], 
            PARAMS['spatial_size'], 
        ])
    elif PARAMS['data_format'] == 'channels_last':
        inputs = tf.keras.Input([
            PARAMS['frames_per_clip'], 
            PARAMS['spatial_size'], 
            PARAMS['spatial_size'], 
            3,
        ])
    outputs = network(inputs)
    return tf.keras.Model(inputs, outputs)







