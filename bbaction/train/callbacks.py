import json
import math
import os

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger, 
    LearningRateScheduler, 
    ModelCheckpoint,
)
# TODO: switch to schedules version once it is moved out of nightly
from tensorflow.keras.experimental import CosineDecay 
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    LearningRateSchedule,
)
from tensorflow.python.keras import backend as K

from utils import get_frozen_params, get_params, get_results_dir, is_file


PARAMS = get_params('Train')
VERSION = PARAMS['version']
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Train',
        version=VERSION,
    )

def savemodel():
    if PARAMS['use_cloud']:
        savemodel_dir = (
            PARAMS['gcs_results'].rstrip('/') + f'/{str(VERSION)}'
        )
    else:
        savemodel_dir = (
            get_results_dir(PARAMS['dataset']) 
            + f'savemodel/{str(VERSION)}'
        )    
    return Checkpoint(savemodel_dir)

def csvlogger(restore=False):
    logs_dir = get_results_dir(PARAMS['dataset']) + 'logs/'
    is_file(
        logs_dir, 
        filename=f'logs_v{VERSION}.csv', 
        restore=restore,
    )
    return CSVLogger(
        filename=logs_dir + f'logs_v{VERSION}.csv',
        append=True,
    )

def lr_schedule_per_step():
    return LRSchedulerPerStep(schedule_per_step)

def schedule_per_step(step, learning_rate):
    if PARAMS['epoch_size'] < 0:
        epoch_size = PARAMS['train_size']
    else:
        epoch_size = PARAMS['epoch_size']
    decay_steps = int(math.ceil(
        PARAMS['decay_epoch'] * epoch_size / PARAMS['batch_per_replica']
    ))

    if PARAMS['use_warmup']:
        if PARAMS['use_half_cosine']:
            if step < decay_steps:
                return WarmupExponentialDecay(
                    initial_learning_rate=PARAMS['lr_per_replica'],
                    decay_steps=decay_steps,
                    decay_rate=PARAMS['decay_rate'],
                    staircase=True,
                    )(step)
            else:
                return CosineDecay(
                    initial_learning_rate=PARAMS['lr_per_replica'],
                    decay_steps=decay_steps,
                    alpha=0.0,
                    )(step)
        else:
            return WarmupExponentialDecay(
                initial_learning_rate=PARAMS['lr_per_replica'],
                decay_steps=decay_steps,
                decay_rate=PARAMS['decay_rate'],
                staircase=True,
                )(step)
    else:
        return ExponentialDecay(
            initial_learning_rate=PARAMS['lr_per_replica'],
            decay_steps=decay_steps,
            decay_rate=PARAMS['decay_rate'],
            staircase=True,
            )(step)


class WarmupExponentialDecay(ExponentialDecay):
    def __init__(
            self, 
            initial_learning_rate, 
            decay_steps, 
            decay_rate, 
            staircase=False, 
            name=None,
        ):
        super(WarmupExponentialDecay, self).__init__(
            initial_learning_rate, 
            decay_steps, 
            decay_rate, 
            staircase=staircase, 
            name=name,
        )
        self.decay_steps = decay_steps

        params = get_params('Train')
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=params['version'],
            )
        else:
            self.params = params
        self.starting_warmup_factor = self.params['warmup_factor']

    def __call__(self, step):
        alpha = float(step) / self.decay_steps
        warmup_factor = self.starting_warmup_factor * (1 - alpha) + alpha
        if step >= self.decay_steps:
            warmup_factor *= 10
        return (
            warmup_factor * super(WarmupExponentialDecay, self).__call__(step)
        )


class LRSchedulerPerStep(LearningRateScheduler):
    def __init__(self, schedule, verbose=0):
        super(LRSchedulerPerStep, self).__init__(schedule, verbose)
        
        params = get_params('Train')
        self.version = params['version']
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=self.version,
            )
        else:
            self.params = params 
        self.strategy = self.params['strategy']
        self.learning_rate = self.params['lr_per_replica']
        self.regularization = self.params['regularization']
        self.weight_decay = self.params['weight_decay']
        self.track_every = self.params['track_every']
        self.steps_per_execution = self.params['steps_per_execution']

        self.step = 1
        results_dir = get_results_dir(self.params['dataset'])
        self.trackers_file = results_dir + 'trackers.json'

    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        super(LRSchedulerPerStep, self).on_epoch_end(epoch, logs)

        with open(self.trackers_file, 'r') as f:
            trackers = json.load(f)
        trackers[f"v{self.version}"]["epoch"] = epoch + 1
        with open(self.trackers_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                trackers, 
                ensure_ascii=False, 
                indent=4,
            ))

    def on_train_batch_begin(self, batch, logs=None):
        super(LRSchedulerPerStep, self).on_epoch_begin(self.step, logs)
        
        if self.regularization == 'weight_decay':
            lr = K.get_value(self.model.optimizer.lr)
            decay = lr / self.learning_rate
            wd = self.weight_decay * decay
            K.set_value(
                self.model.optimizer.weight_decay, 
                K.get_value(wd),
            )

    def on_train_batch_end(self, batch, logs=None):
        if (self.step - 1) % self.track_every == 0:
            with open(self.trackers_file, 'r') as f:
                trackers = json.load(f)
            trackers[f"v{self.version}"]["step"] = self.step
            with open(self.trackers_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(
                    trackers, 
                    ensure_ascii=False, 
                    indent=4,
                ))
        self.step += self.steps_per_execution

class Checkpoint(ModelCheckpoint):
    def __init__(
            self, 
            filepath,
            monitor='val_top_1_acc',
            save_best_only=True,
            mode='max',
            save_weights_only=False,
            verbose=0,
        ):
        super(Checkpoint, self).__init__(
            filepath, 
            monitor=monitor, 
            save_best_only=save_best_only, 
            mode=mode, 
            save_weights_only=save_weights_only,
            verbose=verbose,
        )
        params = get_params('Train')
        self.version = params['version']
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=self.version,
            )
        else:
            self.params = params 
        
        self.best_pck = -1.
        results_dir = get_results_dir(self.params['dataset'])
        self.trackers_file = results_dir + 'trackers.json'

    def on_epoch_end(self, epoch, logs=None):
        super(Checkpoint, self).on_epoch_end(epoch, logs)
        if self.best_pck < self.best:
            self.best_pck = self.best
            with open(self.trackers_file, 'r') as f:
                trackers = json.load(f)
            trackers[f"v{self.version}"]["best"] = self.best_pck
            with open(self.trackers_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(
                    trackers, 
                    ensure_ascii=False, 
                    indent=4,
                ))





