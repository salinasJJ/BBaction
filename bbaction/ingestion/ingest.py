import os
import shutil

from bbdata.download.download import VideoDownloader
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
import tensorflow_io as tfio

from utils import (
    call_bash, 
    force_update,
    freeze_cfg,
    get_data_dir,
    get_frozen_params, 
    get_params,
    is_dir,
)


SEEN_EVERY = 25
MIN_NUM_TS = 0


class DataGenerator():
    def __init__(self, setup=False):
        params = get_params('Ingestion')
        if params['switch']:
            self.params = get_frozen_params(
                'Ingestion', 
                version=params['version'],
            )
        else:
            self.params = params
        self.dataset_dir = get_data_dir(self.params['dataset'])
        if self.params['toy_set']:
            self.data_dir = self.dataset_dir + 'data/toy/'
            self.logs_dir = self.dataset_dir + 'logs/toy/'
            self.tfds_dir = self.dataset_dir + 'tfds/toy/'
            self.records_dir = self.dataset_dir + 'records/toy/'
            self.vid_dir = (
                self.params['vid_dir'] + self.params['dataset'] + '/toy/'
            )
        else:
            self.data_dir = self.dataset_dir + 'data/full/'
            self.logs_dir = self.dataset_dir + 'logs/full/'
            self.tfds_dir = self.dataset_dir + 'tfds/full/'
            self.records_dir = self.dataset_dir + 'records/full/'
            self.vid_dir = (
                self.params['vid_dir'] + self.params['dataset'] + '/full/'
            )
        
        self.count = tf.zeros([], dtype=tf.int64)
        self.train_unusable = tf.cast(
            self.params['train_unusable'],
            tf.int64,
        )
        self.validate_unusable = tf.cast(
            self.params['validate_unusable'],
            tf.int64,
        )

        tf.debugging.assert_less_equal(
            self.params['val_clips_per_vid'],
            self.params['test_clips_per_vid'] ,
            message=(
                "'val_clips_per_vid' should be less than or equal to "
                "'test_clips_per_vid'"
            )
        )   
        tf.debugging.assert_less_equal(
            self.params['fps'],
            self.params['download_fps'],
            message="'fps' should be less than or equal to 'download_fps'"
        )

        self.downloader = VideoDownloader({
            'params':params,
            'dataset_dir':self.dataset_dir,
        })
        if setup:
            self.downloader.get_data()
            self.downloader.setup()

    def generate(self):
        self.generate_datasets()

        freeze_cfg(version=self.params['version'])
        
        if self.params['use_records']:
            shutil.rmtree(self.dataset_dir + 'tfds/')
        else:
            shutil.rmtree(
                self.dataset_dir + 'records/', 
                ignore_errors=True,
            )
            self.generate_test_dataset()
              
    def generate_datasets(self):
        self._set_labels()
        self._set_num_cols()

        i = self.params['loop_it']
        for split in ['train', 'validate']:
            print(f"Generating '{split}' dataset...")           
            wc_file = self.data_dir + split + 'WC.csv'
            csv_file = self.data_dir + split + '.csv'
            if not os.path.isfile(wc_file):
                continue
            ts_file = self.data_dir + 'timestamps.csv'
            if self.params['use_records']:
                is_dir(self.records_dir + split)
            total = call_bash(
                command = f"wc -l < {csv_file}",
            )
            total = int(total.decode('utf-8').strip("\n")) - 1

            while os.path.isfile(wc_file):
                self.downloader.get_videos(split)
                temp_ds = self._csv_to_tfds(ts_file)
                temp_ds = self._map_dataset(temp_ds)

                i += 1
                force_update({'loop_it':i})
                if i == 1:
                    tf.data.experimental.save(
                        temp_ds, 
                        self.tfds_dir + f"{split}_{i}",
                    )

                    if self.params['use_records']:
                        self._create_records(temp_ds, split)
                else:
                    ds = tf.data.experimental.load(
                        path=self.tfds_dir + f"{split}_{i-1}",
                        element_spec=DataGenerator._get_element_spec(
                            self.num_ts_cols,
                        ),
                    )
                    ds = ds.concatenate(temp_ds)
                    tf.data.experimental.save(
                        ds, 
                        self.tfds_dir + f"{split}_{i}"
                    )
                    if self.params['use_records']:
                        self._create_records(ds, split)

                    shutil.rmtree(self.tfds_dir + f"{split}_{i-1}")
                os.remove(ts_file)

                seen = self.params['download_batch'] * i
                if seen % SEEN_EVERY == 0 and seen <= total: 
                    print(f"{seen}/{total}")

            if self.params['use_records']:
                ds = tf.data.experimental.load(
                    path=self.tfds_dir + f"{split}_{i}",
                    element_spec=DataGenerator._get_element_spec(
                        self.num_ts_cols,
                    ),
                )
                self._create_records(ds, split, last_record=True)
                self._set_split_size(split)

            if os.path.isdir(self.tfds_dir + f"{split}_{i}"):
                os.rename(
                    self.tfds_dir + f"{split}_{i}", 
                    self.tfds_dir + f"{split}"
                )
            i = 0
            force_update({'loop_it':i})
            print(
                f"Dataset saved to: {self.tfds_dir}{split}\n"
            )
        if self.params['use_records']:
            if self.params['delete_videos']:
                shutil.rmtree(self.params['vid_dir'] + self.params['dataset']) 
    
    def generate_test_dataset(self):
        val_tables = self._load_dataset('validate').enumerate()

        print("Generating 'test' dataset...")
        test_tables = self._map_test_dataset(val_tables)

        tf.data.experimental.save(
            test_tables, 
            self.tfds_dir + 'test',
        )
        print(f"Dataset saved to: {self.tfds_dir}test\n")

    def _set_labels(self):
        labels = tf.TensorArray(
            tf.string, 
            size=0, 
            dynamic_size=True, 
        )
        with open(self.data_dir + 'labels.txt', 'r') as f:
            for e, l in enumerate(f):
                labels = labels.write(e, l.rstrip('\n'))
            force_update({'num_labels':e + 1})
        self.labels = labels.stack()

    def _set_num_cols(self):
        num_cols = call_bash(
            command = (
                f"awk -F, '{{ print NF; exit }}' {self.data_dir + 'train.csv'}"
            )
        )
        self.num_cols = int(num_cols.decode('utf-8').strip("\n"))
        self.num_ts_cols = (
            self.params['download_fps'] * self.params['time_interval']
        )
    
    def _set_split_size(self, split):
        records = self._get_record_names(split)
        records = sorted(
            records, 
            key=lambda r: int(r.split(f'{split}_')[1].split('.tf')[0]),
        )
        total_records = len(records)

        for enum_0, _ in tf.data.TFRecordDataset(records[0]).enumerate():
            pass
        if total_records > 1:
            for enum_N, _ in tf.data.TFRecordDataset(records[-1]).enumerate():
                pass
            size = ((total_records - 1) * (enum_0 + 1)) + (enum_N + 1)
        else:
            size = enum_0 + 1
        
        if split == 'train':
            size -= self.train_unusable
        else:
            size -= self.validate_unusable

        force_update({f"{split}_size":size.numpy()})
    
    def _csv_to_tfds(self, ts_file):
        field_dtypes = [tf.string for _ in range(self.num_cols)]
        ts_dtypes = [tf.int32 for _ in range(self.num_ts_cols)]

        return tf.data.experimental.CsvDataset(
            ts_file,
            field_dtypes + ts_dtypes,
            exclude_cols=[self.num_cols + self.num_ts_cols],
        )

    def _map_dataset(self, dataset):
        return dataset.map(
            self._get_components, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
    
    def _map_test_dataset(self, dataset):
        return dataset.map(
            self._get_test_components, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
    
    def _get_components(self, *element):
        path = tf.strings.join([
            element[0], '/', 
            element[4], '_', 
            element[1], 
            '.mp4'
        ])
        label = tf.squeeze(
            tf.cast(tf.where(element[0] == self.labels), tf.int32)
        )
        timestamps = tf.stack(element[self.num_cols:])
        identifier = tf.cast(-1, tf.int32)
        return path, label, timestamps, identifier

    def _get_test_components(self, enum, elements):
        return self._get_identifier(enum, elements)
       
    def _create_records(
            self, 
            dataset, 
            split, 
            last_record=False,
        ):
        downloaded = call_bash(
            command = f"wc -l < {self.logs_dir + split + '/downloaded.txt'}"
        )
        downloaded = int(downloaded.decode('utf-8').strip("\n"))

        record_num = call_bash(
            command = (
                f"ls {self.records_dir + split} "
                f"| grep '^{split}' "
                f"| wc -l"
            ),
        )
        record_num = int(record_num.decode('utf-8').strip("\n"))

        videos_per_record = self._get_videos_per_record(split)
        skip = videos_per_record * record_num
        take = videos_per_record

        if last_record and downloaded % videos_per_record != 0:
            threshold = videos_per_record * record_num
        else:
            threshold = videos_per_record * (record_num + 1)

        if downloaded >= threshold:
            self._write_to_record(
                dataset.skip(skip).take(take), 
                split,
                record_num,
            )
    
    def _write_to_record(
            self, 
            dataset, 
            split, 
            record_num,
        ):
        clips_per_vid = self._get_clips_per_vid(split)
        dataset = self._generate_element(
            dataset, 
            clips_per_vid,
            element='frames'
        )

        paths = []
        record = self.records_dir + split + f'/{split}_{record_num}.tfrecord'
        with tf.io.TFRecordWriter(record) as writer:
            starting_id = record_num * self._get_videos_per_record(split)
            for frames, path, label, timestamp, identifier in dataset:                
                paths.append(path.numpy().decode('utf-8'))
                identifier = identifier + starting_id

                example = self._serialize_example(
                    frames,
                    label,
                    timestamp,
                    identifier,
                )
                writer.write(example)

                if timestamp < 0:
                    if split == 'train':
                        self.train_unusable += 1
                        unusable = self.train_unusable.numpy()
                    elif split == 'validate':
                        self.validate_unusable += 1
                        unusable = self.validate_unusable.numpy()
                        
                    force_update({
                        f'{split}_unusable':unusable,
                    })

        paths = list(set(paths))
        if self.params['delete_videos']:
            for p in paths:
                os.remove(self.vid_dir + p)

    def _serialize_example(
            self,
            frames, 
            label,
            timestamp,
            identifier,
        ):
        feature = {
            'frames':DataGenerator._bytes_list_feature(frames),
            'label':DataGenerator._int64_feature(label),
            'timestamp':DataGenerator._int64_feature(timestamp),
            'identifier':DataGenerator._int64_feature(identifier),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        return example.SerializeToString()
    
    @staticmethod
    def _bytes_list_feature(value):
        value = [v.numpy() for v in value]
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=value),
        )

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]),
        )

    def _get_videos_per_record(self, split):
        if split == 'train':
            videos_per_record = self.params['videos_per_record']
        else:
            videos_per_record = self.params['videos_per_test_record']
        return videos_per_record

    def _get_clips_per_vid(self, split):
        if split == 'train':
            clips_per_vid = tf.cast(
                self.params['train_clips_per_vid'], 
                tf.int64,
            )
        elif split == 'validate':
            clips_per_vid = tf.cast(
                tf.math.maximum(
                    self.params['val_clips_per_vid'],
                    self.params['test_clips_per_vid'],
                ),
                tf.int64,
            )
        return clips_per_vid

    def _generate_element(
            self, 
            dataset, 
            clips_per_vid,
            element="frames"
        ):
        if element == "frames":
            ds = self._map_frames(dataset)
        elif element == "timestamp":
            ds = self._map_timestamps(dataset)

        self.count += 1
        if self.count < clips_per_vid:
            ds = ds.concatenate(
                self._generate_element(
                    dataset, 
                    clips_per_vid,
                    element=element,
                ),
            )
        else:
            ds = ds
        self.count -= self.count  
        return ds
    
    def _map_frames(self, dataset):
        ds = dataset.enumerate()
        ds = ds.map(
            self._get_identifier,
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = self._map_timestamps(ds)
        ds = ds.map(
            self._get_video, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.map(
            self._get_frames, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        return ds

    def _map_timestamps(self, dataset):
        return dataset.map(
            self._get_timestamp, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )

    def _get_identifier(self, enum, elements):
        path, label, timestamp, identifier = elements
        identifier = tf.cast(enum, tf.int32)
        return path, label, timestamp, identifier
 
    def _get_timestamp(
            self, 
            path, 
            label, 
            timestamps, 
            identifier,
        ):
        step = self.params['download_fps'] // self.params['fps']
        last_idx = tf.where(timestamps != -1)[-1][0]
        last_idx = last_idx - ((self.params['frames_per_clip'] * step) - 1)
        if last_idx < 0:
            total = -tf.ones([], tf.int64)
        else:
            timestamps = timestamps[:last_idx+step:step]
            tf.debugging.assert_non_negative(
                timestamps, 
                message="Negative values found.",
            )
            total = tf.shape(timestamps, tf.int64)[0]

        if total < 0:
            timestamp = tf.cast(total, tf.int32)
        else:
            timestamp = -tf.ones([], tf.int32)
            if tf.strings.regex_full_match(path, ".*/train_.*"):
                clips_per_vid = self._get_clips_per_vid('train')
                if total >= clips_per_vid:
                    idx_range = total // clips_per_vid
                    idx_start = idx_range * self.count
                    indices = tf.range(
                        start=idx_start, 
                        limit=idx_start+idx_range,
                    )
                else:
                    indices = tf.range(
                        start=0, 
                        limit=total,
                    )
                timestamp = tf.cast(
                    tf.random.shuffle(indices)[0],
                    tf.int32,
                )
            elif tf.strings.regex_full_match(path, ".*/validate_.*"):
                clips_per_vid = self._get_clips_per_vid('validate')
                if total >= clips_per_vid:
                    idx_range = total // clips_per_vid
                    timestamp = tf.cast(
                        idx_range * self.count,
                        tf.int32,
                    )
                else:
                    idx_range = total / clips_per_vid
                    timestamp = tf.cast(
                        tf.math.round(
                            idx_range * tf.cast(self.count, tf.float64)
                        ),
                        tf.int32,
                    )
        return path, label, timestamp, identifier

    def _get_video(
            self, 
            path, 
            label, 
            timestamp, 
            identifier,
        ):
        raw = tf.io.read_file(self.vid_dir + path)
        video = tfio.experimental.ffmpeg.decode_video(raw)
        return video, path, label, timestamp, identifier

    def _get_frames(
            self,
            video, 
            path, 
            label, 
            timestamp, 
            identifier,
        ):
        frames = tf.TensorArray(
            tf.string, 
            size=self.params['frames_per_clip'], 
            dynamic_size=False,
        )
        for f in tf.range(self.params['frames_per_clip']):
            if timestamp < 0:
                frames = frames.write(
                    f,
                    value=tf.io.encode_jpeg(
                        tf.zeros([1,1,1], tf.uint8),
                    )
                )
            else:
                frames = frames.write(
                    f,
                    value=tf.io.encode_jpeg(
                        video[timestamp+f,...],
                        format='rgb',
                    )
                )  
        frames = frames.stack()
        return frames, path, label, timestamp, identifier

    def load_datasets(self):
        train_dataset = self._load_dataset('train')
        train_dataset = self._generate_element(
            train_dataset, 
            self.params['train_clips_per_vid'],
            element='timestamp'
        )
        val_dataset = self._load_dataset('validate')
        val_dataset = self._generate_element(
            val_dataset, 
            self.params['val_clips_per_vid'],
            element='timestamp'
        )
        return train_dataset, val_dataset
      
    def load_test_dataset(self):
        test_dataset = self._load_dataset('test')
        return self._generate_element(
            test_dataset, 
            self.params['test_clips_per_vid'],
            element='timestamp'
        )

    def _load_dataset(self, split):
        num_ts_cols = self.params['download_fps'] * self.params['time_interval']
        return tf.data.experimental.load(
            path=self.tfds_dir + split,
            element_spec=DataGenerator._get_element_spec(num_ts_cols),
        )

    @staticmethod
    def _get_element_spec(num_ts_cols):
        return (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(num_ts_cols,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

    def load_records(self):
        train_records = self._load_records('train')
        val_records = self._load_records('validate')
        return train_records, val_records

    def load_test_records(self):
        return self._load_records('test')
    
    def _load_records(self, split):
        records = self._get_record_names(split)
        ds = tf.data.Dataset.from_tensor_slices(records)
        if split == 'validate':
            ds = ds.interleave(
                tf.data.TFRecordDataset,
                cycle_length=self.params['validate_size'],
                block_length=1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=True,
            )
            return ds.take((
                self.params['validate_size'] 
                // self.params['test_clips_per_vid']
                * self.params['val_clips_per_vid']
            ))
        else:
            ds = ds.shuffle(
                buffer_size=tf.cast(
                    tf.shape(records)[0], 
                    tf.int64,
                ),
            )
            ds = ds.repeat()
            return ds.interleave(
                tf.data.TFRecordDataset,
                cycle_length=self.params['interleave_num'],
                block_length=self.params['interleave_block'],
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False,
            )
    
    def _get_record_names(self, split):
        if split == 'test':
            split = 'validate'
        if self.params['use_cloud']:
            records = tf.io.gfile.glob((
                self.params['gcs_data'].rstrip('/') 
                + '/'
                + split
                + f'/{split}_*.tfrecord'
            ))
        else:
            records = tf.io.gfile.glob(
                self.records_dir + split + f'/{split}*.tfrecord'
            )
        return records


