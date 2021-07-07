import tensorflow as tf
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from utils import get_frozen_params, get_params


class VidTopKAccuracy(SparseTopKCategoricalAccuracy):
    def __init__(
            self, 
            k=5, 
            total_clips=0,
            name='vid_top_k_accuracy', 
            dtype=None,
        ):
        super(VidTopKAccuracy, self).__init__(
            k=k, 
            name=name, 
            dtype=dtype,
        )
        self.k = k
        self.total_clips = tf.Variable(
            initial_value=total_clips,
            dtype=tf.int32,
            trainable=False,
        )

        params = get_frozen_params(
            'Test', 
            version=get_params('Test')['version'],
        )
        self.num_labels = params['num_labels']
        self.test_clips_per_vid = params['test_clips_per_vid']
        self.batch_per_replica = params['batch_per_replica']
        
        self.total_vids = tf.cast(
            self.total_clips // self.test_clips_per_vid, 
            tf.int32,
        )
        self.preds = tf.Variable(
            initial_value=tf.zeros([self.total_vids, self.num_labels]),
            trainable=False,
        )
        self.labels = tf.Variable(
            initial_value=tf.zeros(
                [self.total_vids, 1],
                dtype=tf.int32,
            ),
            trainable=False,
        )
        self.last_step = tf.constant(0)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_config(self):
        config = {
            'total_clips':self.total_clips,
            'num_labels':self.num_labels,
            'test_clips_per_vid':self.test_clips_per_vid,
            'batch_per_replica':self.batch_per_replica,
        }
        base_config = super(VidTopKAccuracy, self).get_config()    
        return dict(
            list(base_config.items()) + list(config.items())
        )

    @tf.function(experimental_relax_shapes=True)
    def update_state(
            self, 
            y_true, 
            y_pred, 
            sample_weight=None,
        ):
        for i in tf.range(self._get_batch_size()):
            pred = y_pred[i]
            true = y_true[i]
            identifier = tf.cast(
                sample_weight[i], 
                tf.int32,
            )
            self.preds.scatter_nd_add(
                indices=tf.expand_dims(identifier, axis=1), 
                updates=tf.expand_dims(pred, axis=0),
            )
            self.labels.scatter_nd_update(
                indices=tf.expand_dims(identifier, axis=1), 
                updates=tf.expand_dims(true, axis=0),
            )
        if self.last_step == 1:
            super(VidTopKAccuracy, self).update_state(
                self.labels, 
                self.preds / self.test_clips_per_vid,
            )

    def result(self):
        return super(VidTopKAccuracy, self).result()

    def reset_state(self):
        super(VidTopKAccuracy, self).reset_states()

    def _get_batch_size(self):
        if self.total_clips > self.batch_per_replica:
            batch_size = tf.cast(
                self.batch_per_replica, 
                tf.int32,
            )
            self.total_clips.assign_sub(
                tf.cast(self.batch_per_replica, tf.int32)
            )
        else:
            batch_size = self.total_clips
            self.total_clips.assign(tf.cast(0, tf.int32))
            self.last_step = tf.constant(1)
        return batch_size






        