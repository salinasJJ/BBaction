import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import (
    CenterCrop, 
    Rescaling, 
    Resizing,
)
import tensorflow_io as tfio

from utils import get_frozen_params, get_params


MIN_RESIZE = {
    112: 128,
    128: 224,
    224: 256,
    256: 256,
}


class DataPreprocessor():
    def __init__(self):
        params = get_params('Preprocess')
        if params['switch']:
            self.params = get_frozen_params(
                'Preprocess', 
                version=params['version'],
            )
        else:
            self.params = params
        
        if self.params['toy_set']:
            self.vid_dir = (
                self.params['vid_dir'] + self.params['dataset'] + '/toy/'
            )
        else:
            self.vid_dir = (
                self.params['vid_dir'] + self.params['dataset'] + '/full/'
            )
        self.batch_size = tf.cast(
            self.params['batch_per_replica'] * self.params['num_replicas'],
            tf.int64,
        )
        self.random_generator = tf.random.Generator.from_seed(1)

    def read_records(self, train_records, val_records):
        train_record_dataset = self._map_records(train_records)
        val_record_dataset = self._map_records(val_records)
        return train_record_dataset, val_record_dataset

    def read_test_records(self, test_records):
        return self._map_records(test_records)

    def _map_records(self, dataset):
        ds = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._parse_features,
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        return ds

    def _parse_example(self, example):
        feature_description = {
            'frames':tf.io.FixedLenFeature(
                [self.params['frames_per_clip']], 
                tf.string,
            ),
            'label':tf.io.FixedLenFeature([], tf.int64),
            'timestamp':tf.io.FixedLenFeature([], tf.int64),
            'identifier':tf.io.FixedLenFeature([], tf.int64),
        }
        features = tf.io.parse_single_example(example, feature_description)
        return features
  
    def _parse_features(self, features):
        frames = features['frames']
        label = tf.cast(features['label'], tf.int32)
        timestamp = tf.cast(features['timestamp'], tf.int32)
        identifier = tf.cast(features['identifier'], tf.int32)
        return frames, label, timestamp, identifier

    def get_datasets(self, train_tables, val_tables):
        train_dataset = self._map_train_dataset(train_tables)
        validation_dataset = self._map_validation_dataset(val_tables)
        return train_dataset, validation_dataset
    
    def get_test_dataset(self, test_tables):
        return self._map_test_dataset(test_tables)

    def _map_train_dataset(self, dataset):
        if self.params['use_records']:
            buffer_size = self.params['shuffle_size']
        else:
            buffer_size = dataset.cardinality()
        
        ds = dataset.shuffle(
            tf.cast(buffer_size, tf.int64)
        )
        ds = ds.filter(self._unusable)
        ds = ds.map(
            self._parsing_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda c, l, i: (Rescaling(scale=1./255)(c), l),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._train_augmentation_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.batch(
            self.batch_size, 
            drop_remainder=False if self.params['use_records'] else True,
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _map_validation_dataset(self, dataset):
        ds = dataset.filter(self._unusable)
        ds = ds.map(
            self._parsing_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda c, l, i: (Rescaling(scale=1./255)(c), l, i),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._val_augmentation_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda c, l, i: (c, l),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _map_test_dataset(self, dataset):
        ds = dataset.filter(self._unusable)
        ds = ds.map(
            self._parsing_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda c, l, i: (Rescaling(scale=1./255)(c), l, i),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._val_augmentation_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _unusable(
            self,
            source,
            label,
            timestamp,
            identifier,
        ):
        if timestamp < 0:
            return tf.zeros([], tf.bool)
        else:
            return tf.ones([], tf.bool)

    @tf.function
    def _parsing_function(
            self, 
            source, 
            label, 
            timestamp,
            identifier,
        ):
        if self.params['use_records']:
            clip = self._get_record_clip(source)
        else:
            clip = self._get_dataset_clip(source, timestamp)
        return clip, label, identifier

    def _get_record_clip(self, frames):
        clip = tf.TensorArray(
            tf.uint8, 
            size=self.params['frames_per_clip'], 
            dynamic_size=False,
        )
        for f in tf.range(self.params['frames_per_clip']):
            clip = clip.write(
                f, 
                tf.io.decode_jpeg(
                    frames[f],
                )
            )
        return tf.cast(clip.stack(), tf.float32)

    def _get_dataset_clip(self, path, timestamp):
        step = self.params['download_fps'] // self.params['fps']
        start = timestamp
        end = start + (self.params['frames_per_clip'] * step)
        
        raw_video = tf.io.read_file(self.vid_dir + path)

        return tf.cast(
            tfio.experimental.ffmpeg.decode_video(
                raw_video,
                )[start:end:step, ...],
            tf.float32,
        )

    @tf.function
    def _train_augmentation_function(self, clip, label):
        clip = self._random_resizing(clip)
        clip = self._random_flipping(clip)
        clip = self._normalize(clip)
        clip = self._random_cropping(clip)
        clip = self._set_data_format(clip)
        return clip, label

    @tf.function
    def _val_augmentation_function(self, clip, label, identifier):
        clip = self._resize(clip)
        clip = self._normalize(clip)
        clip = self._centered_crop(clip)
        clip = self._set_data_format(clip)
        return clip, label, identifier

    def _random_resizing(self, clip):
        H = tf.cast(tf.shape(clip)[1], tf.float32)
        W = tf.cast(tf.shape(clip)[2], tf.float32)
        max_resize = self.params['shorter_edge']
        rand_num = self.random_generator.uniform(
            shape=[], 
            minval=MIN_RESIZE[self.params['spatial_size']], 
            maxval=max_resize + 1, 
            dtype=tf.int32
        )
        shorter_edge = rand_num if rand_num % 2 == 0 else rand_num - 1

        if H > W:
            scaled_w = tf.cast(shorter_edge, tf.float32)
            scaled_h = tf.round(
                (float(shorter_edge) / W) * H
            )
        elif W > H:
            scaled_h = tf.cast(shorter_edge, tf.float32)
            scaled_w = tf.round(
                (float(shorter_edge) / H) * W
            )
        else:
            scaled_h = H
            scaled_w = W

        return Resizing(
            height=tf.cast(scaled_h, tf.int32), 
            width=tf.cast(scaled_w, tf.int32),
            )(clip)

    def _resize(self, clip):
        return Resizing(
            height=tf.cast(
                MIN_RESIZE[self.params['spatial_size']], 
                tf.int32,
            ),
            width=tf.cast(
                MIN_RESIZE[self.params['spatial_size']], 
                tf.int32,
            ),
            )(clip)

    def _random_flipping(self, clip):
        random_chance = self.random_generator.uniform(
            shape=[], 
            minval=0, 
            maxval=2, 
            dtype=tf.int32
        )
        if random_chance == 0:
            return tf.image.flip_left_right(clip)
        else:
            return clip

    def _normalize(self, clip):
        return(clip - self.params['mean']) / self.params['std']

    def _random_cropping(self, clip):
        return tf.image.random_crop(
            clip, 
            size=[
                tf.shape(clip)[0], 
                self.params['spatial_size'], 
                self.params['spatial_size'], 
                3,
            ],
        )
        
    def _centered_crop(self, clip):
        return CenterCrop(
            height=self.params['spatial_size'], 
            width=self.params['spatial_size'],
            )(clip)

    def _set_data_format(self, clip):
        if self.params['data_format'] == 'channels_first':
            return tf.transpose(
                clip,
                perm=[3,0,1,2],
            )
        elif self.params['data_format'] == 'channels_last':
            return clip


