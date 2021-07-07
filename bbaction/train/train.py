import json
import math
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import tensorflow_addons as tfa

from ingestion.ingest import DataGenerator
from models import model as cnn
from preprocess.preprocess import DataPreprocessor
from train import callbacks
from utils import (
    force_update, 
    freeze_cfg, 
    get_frozen_params, 
    get_params,
    get_results_dir,
    is_file,
    reload_modules,
)


def run(restore=False):
    force_update({'switch':False})
    reload_modules(cnn, callbacks)

    params = get_params('Train')
    version = params['version']
    if restore:
        force_update({'switch':True})
        reload_modules(cnn, callbacks)

        params = get_frozen_params(
            'Train', 
            version=version,
        )

        results_dir = get_results_dir(params['dataset'])
    else:
        freeze_cfg(version=version)
        results_dir = get_results_dir(params['dataset'])
        
        skeleton = {}
        skeleton[f'v{version}'] = {
            'best': 0.0, 
            'epoch': 0, 
            'step': 0,
        }
        is_file(results_dir, 'trackers.json')
        with open(results_dir + 'trackers.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                skeleton, 
                ensure_ascii=False, 
                indent=4,
            ))
    
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
    reload_modules(callbacks)

    if params['mixed_precision'] and params['strategy'] == 'tpu':
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    
    assert params['track_every'] % params['steps_per_execution'] == 0, \
        "'track_every' must be a multiple of 'steps_per_execution'"

    generator = DataGenerator()
    preprocessor = DataPreprocessor()
    if params['use_records']:
        train_records, val_records = generator.load_records()
        train_tables, val_tables = preprocessor.read_records(
            train_records,
            val_records,
        )
    else:
        assert params['strategy'] != 'tpu', \
            "TPUStrategy only supports TFRecords as input"
        if os.path.isdir(params['vid_dir']):
            train_tables, val_tables = generator.load_datasets()
        else:
            print(f"{params['vid_dir']} does not exist")

    train_dataset, val_dataset = preprocessor.get_datasets(
        train_tables, 
        val_tables,
    )
    
    callbacks_list = [
        callbacks.lr_schedule_per_step(),
        callbacks.savemodel(), 
        callbacks.csvlogger(restore),
    ]

    if params['use_records']:
        if params['epoch_size'] < 0:
            train_size = params['train_size']
        else:
            train_size = params['epoch_size']
        steps_per_epoch = train_size // params['batch_per_replica']
        
        validate_size = (
            params['validate_size'] 
            // params['test_clips_per_vid']
            * params['val_clips_per_vid']
        )
        validation_steps = int(math.ceil(
            validate_size / params['batch_per_replica']
        ))
    else:
        steps_per_epoch = None
        validation_steps = None

    if restore:
        if params['use_cloud']:
            savemodel_dir = (
                params['gcs_results'].rstrip('/') + f'/{str(version)}'
            )
        else:
            savemodel_dir = results_dir + f'savemodel/{str(version)}'

        if tf.io.gfile.isdir(savemodel_dir):
            print('Restoring...')
            with open(results_dir + 'trackers.json', 'r') as f:
                trackers = json.load(f)

            callbacks_list[1].best = trackers[f"v{version}"]["best"]
            initial_epoch = int(trackers[f"v{version}"]["epoch"])
            callbacks_list[0].step = int(trackers[f"v{version}"]["step"])

            with strategy.scope():
                model = tf.keras.models.load_model(
                    savemodel_dir, 
                    custom_objects=get_custom_objects(),
                )
                model = get_compiled_model(model, params)
        else:
            print('No SaveModel found, creating a new model from ' \
                  'scratch...')
            initial_epoch = 0
            with strategy.scope():
                model = cnn.get_model()  
                model = get_compiled_model(model, params)
    else:
        print('Creating a new model...')
        initial_epoch = 0
        with strategy.scope():
            model = cnn.get_model()
            model = get_compiled_model(model, params)

    force_update({'switch':False})

    model.fit(
        train_dataset, 
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=params['num_epochs'], 
        callbacks=callbacks_list,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    return model

def get_compiled_model(model, params):
    if params['regularization'] == 'weight_decay':
        optimizer = tfa.optimizers.SGDW(
            learning_rate=params['lr_per_replica'], 
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
            nesterov=params['nesterov'],
        )
    elif params['regularization'] == 'l2':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=params['lr_per_replica'], 
            momentum=params['momentum'],
            nesterov=params['nesterov'],
        )
    else:
        print((
            "Unsupported regularization technique. Please choose between " 
            "'weight_decay' or 'l2'."
        ))

    model.compile(
        optimizer=optimizer, 
        loss=SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
        ),
        metrics=[
            SparseTopKCategoricalAccuracy(
                k=1, 
                name='top_1_acc',
            ),
            SparseTopKCategoricalAccuracy(
                k=params['max_k'], 
                name='top_5_acc',
            ),
        ],
        steps_per_execution=params['steps_per_execution'],
    )
    return model

def get_custom_objects():
    return {
        "LRSchedulerPerStep": callbacks.LRSchedulerPerStep,
        "WarmupExponentialDecay": callbacks.WarmupExponentialDecay,
        "Checkpoint": callbacks.Checkpoint,
    }





