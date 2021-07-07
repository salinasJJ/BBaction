import math
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from ingestion.ingest import DataGenerator
from models import test_model as cnn
from preprocess.preprocess import DataPreprocessor
from test import callbacks
from test.metrics import VidTopKAccuracy
from train.train import get_custom_objects
from utils import (
    force_update, 
    get_frozen_params, 
    get_params,
    get_results_dir,
    reload_modules,
)


def run():
    force_update({'switch':True})
    reload_modules(cnn, callbacks)

    version=get_params('Test')['version']
    params = get_frozen_params(
        'Test', 
        version=version,
    )

    tf.config.run_functions_eagerly(params['is_eager'])

    if params['strategy'] == 'default':
        strategy = tf.distribute.get_strategy()
    elif params['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    elif params['strategy'] == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=params['tpu_address']
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

    force_update({'num_replicas':strategy.num_replicas_in_sync})

    if params['mixed_precision'] and params['strategy'] == 'tpu':
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    generator = DataGenerator()
    preprocessor = DataPreprocessor()
    if params['use_records']:
        test_records = generator.load_test_records()
        test_tables = preprocessor.read_test_records(test_records)
    else:
        assert params['strategy'] != 'tpu', \
            "TPUStrategy only supports TFRecords as input"
        if os.path.isdir(params['vid_dir']):    
            test_tables = generator.load_test_dataset()
        else:
            print(f"{params['vid_dir']} does not exist")
    
    test_dataset = preprocessor.get_test_dataset(test_tables)

    callbacks_list = [
        callbacks.csvlogger(),
    ]

    if params['use_records']:
        total_clips = params['validate_size']
        steps = int(math.ceil(
            total_clips / params['batch_per_replica']
        ))
    else:
        total_clips = test_tables.cardinality().numpy()
        steps = None

    if params['use_cloud']:
        savemodel_dir = (
            params['gcs_results'].rstrip('/') + f'/{str(version)}'
        )
    else:
        savemodel_dir = (
            get_results_dir(params['dataset']) 
            + f'savemodel/{str(version)}'
        )

    if tf.io.gfile.isdir(savemodel_dir):
        with strategy.scope():
            print('Loading model weights...')
            loaded = tf.keras.models.load_model(
                savemodel_dir, 
                custom_objects=get_custom_objects(),
            )

            print('compiling...')
            model = cnn.get_model(loaded)
            model.compile(
                metrics=[
                    SparseTopKCategoricalAccuracy(
                        k=1, 
                        name='clip_top_1_acc',
                    ),
                    SparseTopKCategoricalAccuracy(
                        k=params['max_k'], 
                        name='clip_top_5_acc',
                    ),
                ],
                weighted_metrics=[
                    VidTopKAccuracy(
                        k=1, 
                        total_clips=total_clips,
                        name='vid_top_1_acc',
                    ),
                    VidTopKAccuracy(
                        k=params['max_k'],
                        total_clips=total_clips,
                        name='vid_top_5_acc',
                    ),
                ],
                steps_per_execution=params['steps_per_execution'],
            )

        force_update({'switch':False,})

        model.evaluate(
            test_dataset,
            callbacks=callbacks_list,
            steps=steps,
        )
        return model
    else:
        force_update({'switch':False,})
        print('No SaveModel was found.')
        return

    




        