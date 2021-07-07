import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ingestion.ingest import DataGenerator
from test import test
from train import train
from utils import (
    force_update,
    reload_modules,
    reset_default_params,
    set_cookies_path,
    set_media_directory,
    set_model_version, 
    update_params,
)


def str_to_bool(v):
    if isinstance(v, bool):
       return v
    elif v.lower() in ['true', 't', '1']:
        return True
    elif v.lower() in ['false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(
        description="Video action recognition model.",
    )
    subparsers = parser.add_subparsers(
        dest='subparser_name', 
        help='sub-command help',
    )

    update_parser = subparsers.add_parser(
        'update', 
        help="updates the model parameters",
    )
    update_parser.add_argument(
        '--version', type=int, required=True,
        help="model version number",
    )
    update_parser.add_argument(
        '--vid_dir', type=str, default='/PATH/TO/VID/DIR', 
        help="directory where video files can be found",
    )
    update_parser.add_argument(
        '--dataset', type=str, default='kinetics400',
        help=(
            "one of 'kinetics400', 'kinetics600', 'kinetics700', "
            "'kinetics700_2020, 'HACS', 'actnet100', or 'actnet200'"
        ),
    )
    update_parser.add_argument(
        '--cookies', type=str, default='/PATH/TO/COOKIES/DIR',
        help="cookies to pass to youtube-dl",
    )
    update_parser.add_argument(
        '--use_records', type=str_to_bool, default=False,
        help="whether to generate and use tfrecords or not",
    )
    update_parser.add_argument(
        '--use_cloud', type=str_to_bool, default=False,
        help="whether to retrieve data from a remote GCS location or not",
    )
    update_parser.add_argument(
        '--batch_per_replica', type=int, default=8,
        help="number of samples passed to each replica",
    )
    update_parser.add_argument(
        '--conda_path', type=str, default='none',
        help="absolute path to conda package (if running a conda env)",
    )
    update_parser.add_argument(
        '--conda_env', type=str, default='none',
        help="name of your environment (if running a conda env)",
    )
    update_parser.add_argument(
        '--num_jobs', type=int, default=5,
        help="number of simultaneous jobs to run with GNU Parallel",
    )
    update_parser.add_argument(
        '--toy_set', type=str_to_bool, default=False,
        help="whether to use a smaller dataset to experiment with or not",
    )
    update_parser.add_argument(
        '--toy_samples', type=int, default=100,
        help="number of samples for toy dataset",
    )
    update_parser.add_argument(
        '--download_batch', type=int, default=20,
        help="batch of videos to download on each iteration",
    )
    update_parser.add_argument(
        '--download_fps', type=int, default=30,
        help="frame rate to download each video with",
    )
    update_parser.add_argument(
        '--time_interval', type=int, default=10,
        help="length of video to be downloaded (in seconds)",
    )
    update_parser.add_argument(
        '--shorter_edge', type=int, default=320,
        help="length of frame's shorter side to download at",
    )
    update_parser.add_argument(
        '--use_sampler', type=str_to_bool, default=False,
        help="whether to use a clip sampler or not",
    )
    update_parser.add_argument(
        '--max_duration', type=int, default=300,
        help="max length of video to sample from",
    )
    update_parser.add_argument(
        '--num_samples', type=int, default=10,
        help="total number of sampled clips",
    )
    update_parser.add_argument(
        '--sampling', type=str, default='random',
        help="one of 'random' or 'uniform' sampling",
    )
    update_parser.add_argument(
        '--sample_duration', type=int, default=1,
        help="duration of each sampled clip",
    )
    update_parser.add_argument(
        '--videos_per_record', type=int, default=125,
        help="number of videos per training tfrecord",
    )
    update_parser.add_argument(
        '--videos_per_test_record', type=int, default=50,
        help="number of videos per testing tfrecord",
    )
    update_parser.add_argument(
        '--fps', type=int, default=15,
        help="frame rate to preprocess each video clip with",
    )
    update_parser.add_argument(
        '--frames_per_clip', type=int, default=8,
        help="one of 1, 2, 4, or 8*N (N=1,2,3,...) frames per clip",
    )
    update_parser.add_argument(
        '--train_clips_per_vid', type=int, default=4,
        help="total number of training clips selected for each unique video",
    )
    update_parser.add_argument(
        '--val_clips_per_vid', type=int, default=1,
        help="total number of validation clips selected for each unique video",
    )
    update_parser.add_argument(
        '--test_clips_per_vid', type=int, default=10,
        help="total number of testing clips selected for each unique video",
    )
    update_parser.add_argument(
        '--delete_videos', type=str_to_bool, default=False,
        help="whether to delete videos after saving files to tfrecords",
    )
    update_parser.add_argument(
        '--interleave_num', type=int, default=2,
        help="number of records to simultaneously interleave",
    )
    update_parser.add_argument(
        '--interleave_block', type=int, default=1,
        help="number of consecutive elements from each record",
    )
    update_parser.add_argument(
        '--gcs_data', type=str, default='',
        help="remote GCS location where tfrecords may be found",
    )
    update_parser.add_argument(
        '--spatial_size', type=int, default=224,
        help="one of 112, 128, 224, or 256 (length used to crop each frame)",
    )
    update_parser.add_argument(
        '--shuffle_size', type=int, default=250,
        help="sample size to shuffle and place into buffer",
    )
    update_parser.add_argument(
        '--mean', type=float, nargs=3, default=[0.43216, 0.394666, 0.37645],
        help="mean values calculated (from dataset) for each channel (rgb)",
    )
    update_parser.add_argument(
        '--std', type=float, nargs=3, default=[0.22803, 0.22145, 0.216989],
        help="std values calculated (from dataset) for each channel (rgb)",
    )
    update_parser.add_argument(
        '--data_format', type=str, default='channels_last',
        help="one of 'channels_last' (NTHWC) or 'channels_first' (NCTHW)",
    )
    update_parser.add_argument(
        '--arch', type=str, default='ip-csn',
        help="one of 'ip-csn' or 'ir-csn' (model architecture to be used)",
    )
    update_parser.add_argument(
        '--depth', type=int, default=50,
        help="one of 26, 50, 101, or 152 (convolutional layers to be used)",
    )
    update_parser.add_argument(
        '--regularization', type=str, default='weight_decay',
        help="one of 'weight_decay' or 'l2' (technique to be used)",
    )
    update_parser.add_argument(
        '--weight_decay', type=float, default=0.0001,
        help="L2 regularization factor",
    )
    update_parser.add_argument(
        '--initializer', type=str, default='he_normal',
        help=(
            "one of 'glorot_normal', 'glorot_uniform', 'he_normal', or "
            "'he_uniform' (used in each layer)"
        ),
    )
    update_parser.add_argument(
        '--use_bias', type=str_to_bool, default=False,
        help="whether to include a bias term with each layer or not",
    )
    update_parser.add_argument(
        '--bn_momentum', type=float, default=0.9,
        help="momentum for moving average used in each batch norm layer",
    )
    update_parser.add_argument(
        '--epsilon', type=float, default=0.001,
        help="value added to batch norm's variance to avoid division by zero",
    )
    update_parser.add_argument(
        '--dropout_rate', type=float, default=0.0,
        help="percentage of units to drop (if used, recommended: 0.2)",
    )
    update_parser.add_argument(
        '--is_eager', type=str_to_bool, default=False,
        help="whether to run tf.functions in eager mode or not",
    )
    update_parser.add_argument(
        '--strategy', type=str, default='default',
        help="one of 'default' or 'mirrored' (distributed training)",
    )
    update_parser.add_argument(
        '--tpu_address', type=str, default='',
        help="remote location of TPU device(s)"
    )
    update_parser.add_argument(
        '--gcs_results', type=str, default='',
        help="remote GCS location where savemodels may be found",
    )
    update_parser.add_argument(
        '--mixed_precision', type=str_to_bool, default=False,
        help="whether to use both 16 and 32 bit floating-point types or not",
    )
    update_parser.add_argument(
        '--num_epochs', type=int, default=45,
        help="number of epochs to train for",
    )
    update_parser.add_argument(
        '--epoch_size', type=int, default=-1,
        help="total number of clips to use per epoch",
    )
    update_parser.add_argument(
        '--steps_per_execution', type=int, default=1,
        help="number of batches to run through each tf.function",
    )
    update_parser.add_argument(
        '--use_warmup', type=str_to_bool, default=True,
        help="whether to use model warming-up technique or not",
    )
    update_parser.add_argument(
        '--warmup_factor', type=float, default=0.00001,
        help="starting decay factor to use during model warm up",
    )
    update_parser.add_argument(
        '--use_half_cosine', type=str_to_bool, default=True,
        help="whether to use half cosine decay (post warmup) or not",
    )
    update_parser.add_argument(
        '--decay_epoch', type=int, default=10,
        help="how often (i.e. every ith epoch) to apply decay rate",
    )
    update_parser.add_argument(
        '--decay_rate', type=float, default=0.1,
        help="factor to decay learning rate by",
    )
    update_parser.add_argument(
        '--lr_per_replica', type=float, default=0.01,
        help="initial learning rate per device",
    )
    update_parser.add_argument(
        '--momentum', type=float, default=0.9,
        help="value to accelerate gradient descent by",
    )
    update_parser.add_argument(
        '--nesterov', type=str_to_bool, default=True,
        help="whether to apply Nesterov momentum or not",
    )
    update_parser.add_argument(
        '--track_every', type=int, default=32,
        help="how often to save current iteration/step for use with LR schedule",
    )
    update_parser.add_argument(
        '--max_k', type=int, default=5,
        help="total number of top predictions to use in calculating accuracy",
    )

    force_parser = subparsers.add_parser(
        'force',
        help="updates hidden params",
    )
    force_parser.add_argument(
        '--train_size', type=int, default=0,
        help="number of elements in the train dataset",
    )
    force_parser.add_argument(
        '--validate_size', type=int, default=0,
        help="number of elemenets in the validation dataset",
    )
    force_parser.add_argument(
        '--train_unusable', type=int, default=0,
        help="number of invalid elements in the train dataset",
    )
    force_parser.add_argument(
        '--validate_unusable', type=int, default=0,
        help="number of invalid elements in the validation dataset",
    )
    force_parser.add_argument(
        '--num_labels', type=int, default=0,
        help="total number of labels",
    )

    reset_parser = subparsers.add_parser(
        'reset',
        help="resets params to default values",
    )
    reset_parser.add_argument(
        '--defaults', type=str, default='tran',
        help="set of default params",
    )

    ingest_parser = subparsers.add_parser(
        'ingest', 
        help="retrieves data and generates datasets from scratch",
    )
    ingest_parser.add_argument(
        '--setup', action='store_true',
        help="whether to run setup script or not",
    )
    ingest_parser.add_argument(
        '--generate', action='store_true',
        help="whether to generate the datasets or not"
    )

    train_parser = subparsers.add_parser(
        'train',
        help="trains the model",
    )
    train_parser.add_argument(
        '--restore', action='store_true',
        help="whether to restore a saved model or not",
    )

    test_parser = subparsers.add_parser(
        'test',
        help="evaluates a trained model",
    )

    args = vars(parser.parse_args())

    if args['subparser_name'] == 'reset':
        reset_default_params(args['defaults'])
        print('Params reset to default values.')

    elif args['subparser_name'] == 'force':
        args.pop('subparser_name')
        force_update(args)

    elif args['subparser_name'] == 'update':
        args.pop('subparser_name')
        set_model_version(args.pop('version'))
        set_media_directory(args.pop('vid_dir'))
        set_cookies_path(args.pop('cookies'))
        update_params(args)
        print('Update complete.')
    
    elif args['subparser_name'] == 'ingest':
        if args['setup']:
            generator = DataGenerator(setup=True)
        else:
            generator = DataGenerator()
        if args['generate']:
            generator.generate()
        else:
            pass

    elif args['subparser_name'] == 'train':
        reload_modules(train)
        train.run(restore=args['restore'])

    elif args['subparser_name'] == 'test':
        reload_modules(test)
        test.run()
    
if __name__ == '__main__':
    main()





