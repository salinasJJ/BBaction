from configparser import ConfigParser, ExtendedInterpolation
from importlib import reload
import os
import shutil
import subprocess
import sys


BB_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
CFG = BB_DIR + 'configs/'
MAIN = 'tran'
KINETICS = 'https://storage.googleapis.com/deepmind-media/Datasets/'
HACS = 'http://hacs.csail.mit.edu/dataset/'
ACTNET = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/'
DATASETS = {
    'kinetics400':KINETICS+'kinetics400.tar.gz',
    'kinetics600':KINETICS+'kinetics600.tar.gz',
    'kinetics700':KINETICS+'kinetics700.tar.gz',
    'kinetics700_2020':KINETICS+'kinetics700_2020.tar.gz',
    'hacs':HACS+'HACS_v1.1.1.zip',
    'actnet100':ACTNET+'activity_net.v1-2.min.json',
    'actnet200':ACTNET+'activity_net.v1-3.min.json',
}
SECTIONS = [
    'Ingestion', 
    'Preprocess', 
    'Models',
    'Train',
    'Test',
]
REQUIRED = [
    'version',
    'vid_dir',
    'cookies',
]
TYPES = [
    'string', 
    'integer', 
    'float', 
    'boolean', 
    'list',
]
DEFAULTS = {
    'tran':{
        'string':{
            'vid_dir':'/PATH/TO/VID/DIR',
            'cookies':'/PATH/TO/COOKIES/DIR',
            'strategy':'default',
            'tpu_address':'',
            'gcs_results':'',
            'dataset':'kinetics400',
            'conda_path':'none',
            'conda_env':'none',
            'sampling':'random',
            'gcs_data':'',
            'data_format':'channels_last',
            'arch':'ip-csn',
            'regularization':'weight_decay',
            'initializer':'he_normal',
        },
        'integer':{
            'version':0,
            'batch_per_replica':8,
            'num_replicas':1,
            'train_size':0,
            'validate_size':0,
            'train_unusable':0,
            'validate_unusable':0,
            'num_labels':0,
            'loop_it':0,
            'num_jobs':5,
            'toy_samples':100,
            'download_batch':20,
            'download_fps':30,
            'time_interval':10,
            'shorter_edge':320,
            'max_duration':-1,
            'num_samples':10,
            'sample_duration':1,
            'videos_per_record':125,
            'videos_per_test_record':50,
            'fps': 15,
            'frames_per_clip': 8,
            'train_clips_per_vid': 4,
            'val_clips_per_vid': 1,
            'test_clips_per_vid': 10,
            'interleave_num':2,
            'interleave_block':1,
            'spatial_size':224,
            'shuffle_size':250,
            'depth':50,
            'num_epochs':45,
            'epoch_size':-1,
            'steps_per_execution':1,
            'decay_epoch':10,
            'track_every':32,
            'max_k':5,
        },
        'float':{
            'bn_momentum':0.9,
            'epsilon':0.001,
            'dropout_rate':0.0,
            'warmup_factor':0.00001,
            'decay_rate':0.1,
            'lr_per_replica':0.01,
            'momentum':0.9,
            'weight_decay':0.0001,
        },
        'boolean':{
            'switch':False,
            'use_records':False,
            'use_cloud':False,
            'toy_set':False,
            'use_sampler':False,
            'delete_videos':False,
            'use_bias':False,
            'is_eager':False,
            'mixed_precision':False,
            'use_warmup':True,
            'use_half_cosine':True,
            'nesterov':True, 
        },
        'list':{
            'mean':[0.43216, 0.394666, 0.37645],
            'std':[0.22803, 0.22145, 0.216989],
        },   
    },
}
IGNORE = [
    'switch',
    'num_replicas',
    'train_size',
    'validate_size',
    'train_unusable',
    'validate_unusable',
    'num_labels',
    'loop_it',
]
DUPLICATES = {
    'DEFAULT':[],
    'Ingestion':[],
    'Preprocess':[
        'toy_set',
        'download_fps',
        'shorter_edge',
        'fps',
        'frames_per_clip',
    ],
    'Models':[
        'frames_per_clip',
        'spatial_size',
        'data_format',
    ],
    'Train':[
        'val_clips_per_vid'
        'test_clips_per_vid'
        'regularization',
        'weight_decay',
    ],
    'Test':[
        'test_clips_per_vid',
        'is_eager',
        'strategy',
        'tpu_address',
        'gcs_results',
        'mixed_precision',
        'max_k',
    ],
}
FILTERS = {
    'strategy':[
        'default',
        'mirrored',
        'tpu',
    ],
    'dataset':[
        'kinetics400',
        'kinetics600',
        'kinetics700',
        'kinetics700_2020',
        'hacs',
        'actnet100',
        'actnet200',
    ],
    'sampling':[
        'random',
        'uniform',
    ],
    'arch':[
        'ip-csn',
        'ir-csn',
    ],
    'initializer':[
        'glorot_normal', 
        'glorot_uniform', 
        'he_normal', 
        'he_uniform',
    ],
    'depth':[26, 50, 101, 152],
    'spatial_size':[112, 128, 224, 256],
    'frames_per_clip':[1, 2, 4],
    'data_format':[
        'channels_last',
        'channels_first',
    ],
    'regularization':[
        'weight_decay',
        'l2',
    ]
}


def get_params(*args, cfg=CFG+'config.cfg'):
    config = ConfigParser(
        interpolation=ExtendedInterpolation(),
    )
    config.read(cfg)
    params = {}
    for section in args:
        assert isinstance(section, str), f"'{section}' must be a string"
        if section.capitalize() in SECTIONS:
            for option in config.options(section):
                params[option] = eval(config.get(
                    section, 
                    option,
                ))
        else:
            print((
                f"Section '{section}' was not found in the configuration file."
            ))
            print(f"Available sections: {SECTIONS}")
    return params

def update_config(*args, cfg=CFG+'config.cfg'):
    config = ConfigParser(
        interpolation=ExtendedInterpolation(),
    )
    config.read(cfg)
    for arg in args:
        config.set(
            arg[0], 
            arg[1], 
            arg[2],
        )
    with open(cfg, 'w') as f:
        config.write(f)

def set_model_version(version):
    assert isinstance(version, int), f"'{version}' must be an integer"

    update_params({
        REQUIRED[0]:version,
    })

def set_media_directory(media_dir):
    assert isinstance(media_dir, str), f"'{media_dir}' must be a string"

    update_config((
        'DEFAULT', 
        REQUIRED[1], 
        '\'' + media_dir.rstrip('/') + '/\'',
    ))

def set_cookies_path(path):
    assert isinstance(path, str), f"'{path}' must be a string"

    update_config((
        SECTIONS[0], 
        REQUIRED[2], 
        '\'' + path.rstrip('/') + '\'',
    ))

def update_params(param_dict, cfg=CFG+'config.cfg'):
    assert isinstance(param_dict, dict), f"'{param_dict}' must be a dictionary"
    assert isinstance(cfg, str), f"'{cfg}' must be a string"

    force_update({'switch':False})

    config = ConfigParser(
        interpolation=ExtendedInterpolation()
    )
    config.read(cfg)

    param_dict = filter_params(param_dict)
    for param, value in param_dict.items():
        if param in config.defaults().keys():
            conditional_update(
                'DEFAULT',
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[0], param):
            conditional_update(
                SECTIONS[0],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[1], param):
            conditional_update(
                SECTIONS[1],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[2], param):
            conditional_update(
                SECTIONS[2],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[3], param):
            conditional_update(
                SECTIONS[3],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[4], param):
            conditional_update(
                SECTIONS[4],
                param,
                value,
                cfg=cfg,
            ) 

def filter_params(param_dict):
    param_list = get_param_list()

    for param, value in param_dict.copy().items():
        if param not in param_list:
            print(f"Unknown param: '{param}'")
            param_dict.pop(param)
            continue

        if not type_check(param, value):
            type_message(param, value)
            param_dict.pop(param)
            continue

        if param in IGNORE:
            print(f"'{param}' is not an adjustable parameter.\n")
            param_dict.pop(param)
        elif param == 'initializer':
            if value not in FILTERS[param]:
                print(
                    f"Recommended '{param}' values include: {FILTERS[param]}."
                )
                print((
                    "Please check with the tensorflow documentation first to " 
                    "determine if your requested initializer is supported. If " 
                    "not, you may receive an error at runtime."
                ))
                print((
                    'https://www.tensorflow.org/api_docs/python/tf/keras/'
                    'initializers\n'
                ))
        elif param == 'frames_per_clip':
            if (value < 8 and
                value not in FILTERS[param] or
                value >= 8 and
                value % 8 != 0):
                    
                print((
                    f"Supported '{param}' values include: {FILTERS[param]} "
                    f"or any multiple of 8. "
                ))
                print(f"'{param}' was not updated.\n")
                param_dict.pop(param)
        elif param == 'delete_videos':
            if value == True:
                print(f"Warning: param '{param}' is set to 'True'.")
                print((
                    f"If 'use_records' is set to 'False', '{param}' will be "
                    f"ignored. But if 'use_records' is also set to 'True', "
                    f"this will lead to every video being deleted after being "
                    f"saved to a tfrecord." 
                ))
                print("Please make sure this is the desired outcome.\n")
        elif param == 'strategy':
            generic_filter(param_dict, param)
        elif param == 'dataset':
            generic_filter(param_dict, param)  
        elif param == 'sampling':
            generic_filter(param_dict, param)     
        elif param == 'arch':
            generic_filter(param_dict, param)
        elif param == 'depth':
            generic_filter(param_dict, param)
        elif param == 'spatial_size':
            generic_filter(param_dict, param)
        elif param == 'data_format':
            generic_filter(param_dict, param)
        elif param == 'regularization':
            generic_filter(param_dict, param)
    return param_dict

def conditional_update(
        section, 
        param,
        value,
        cfg=CFG+'config.cfg',
    ):
    if param in DUPLICATES[section]:
        pass
    elif param in DEFAULTS[MAIN]['string']:
        update_config(
            (section, f'{param}', f"'{value}'"),
            cfg=cfg,
        )
    else:
        update_config(
            (section, f'{param}', f'{value}'),
            cfg=cfg,
        )

def get_param_list():
    param_list = []
    for t in TYPES:
        param_list += [k for k in DEFAULTS[MAIN][t]]
    return param_list

def type_check(param, value):
    if isinstance(value, bool):
        if param in DEFAULTS[MAIN]['boolean']:
            return True
    elif isinstance(value, list):
        if param in DEFAULTS[MAIN]['list']:
            list_type = type(
                DEFAULTS[MAIN]['list'][param][0]
            )
            if all(isinstance(elem, list_type) for elem in value):
                return True        
    elif isinstance(value, int):
        if param in DEFAULTS[MAIN]['integer']:
            return True
    elif isinstance(value, float):
        if param in DEFAULTS[MAIN]['float']:
            return True
    elif isinstance(value, str):
        if param in DEFAULTS[MAIN]['string']:
            return True
    return False

def type_message(param, value):
    if isinstance(value, list):
        list_type = type(
            DEFAULTS[MAIN]['list'][param][0]
            ).__name__
        print((
            f"'{param}' must be of type 'list' containing elements of type "
            f"'{list_type}'"
        ))
        print(f"'{param}' was not updated.\n")
    else:
        param_type = get_type(param)
        print(f"'{param}' must be of type '{param_type}'")
        print(f"'{param}' was not updated.\n")
    
def generic_filter(param_dict, param):
    if param_dict[param] not in FILTERS[param]:
        print(f"Supported '{param}' values include: {FILTERS[param]}.")
        print(f"'{param}' was not updated.\n")
        param_dict.pop(param)  

def get_type(param):
    for k, v in DEFAULTS[MAIN].items():
        if param in v:
            return k

def force_update(param_dict):
    config = ConfigParser(
        interpolation=ExtendedInterpolation(),
    )
    config.read(CFG + 'config.cfg')

    for param in param_dict:
        if param in config.defaults().keys():
            if param in IGNORE:
                update_config((
                    'DEFAULT', 
                    f'{param}', 
                    f'{param_dict[param]}',
                ))

def view_config(version='current'):
    if version == 'current':
        params = get_params(*SECTIONS)
    elif isinstance(version, int):
        try:
            params = get_frozen_params(
                *SECTIONS,
                version=version,
            )
        except:
            return f"No config file for version '{version}' exists."
    else:
        return 'Invalid entry.'
    
    for ignore in IGNORE:
        params.pop(ignore)
    return params

def reset_default_params(defaults=MAIN):
    default_dict = {}
    ignore_dict = {}
    for k, v in DEFAULTS[defaults].items():
        for nested_k, nested_v in v.items():
            if nested_k not in IGNORE:
                default_dict[nested_k] = nested_v
            else:
                ignore_dict[nested_k] = nested_v
    update_params(default_dict)
    force_update(ignore_dict)

def freeze_cfg(version):
    shutil.copyfile(
        CFG + 'config.cfg', 
        CFG + f'freeze_v{version}.cfg',
    )

def get_frozen_params(*section, version=0):
    return get_params(
        *section, 
        cfg=CFG + f'freeze_v{version}.cfg',
    )

def update_frozen_params(param_dict, version=0):
    assert isinstance(param_dict, dict), f"'{param_dict}' must be a dictionary"
    assert isinstance(version, int), f"'{version}' must be an integer"

    update_params(
        param_dict,
        cfg=CFG + f'freeze_v{version}.cfg',
    )
    
def reload_modules(*args):
    for arg in args:
        reload(arg)

def is_file(
        path, 
        filename, 
        restore=False,
    ):
    if os.path.isdir(path):
        if os.path.isfile(path + filename):
            if restore == False:
                open(path + filename, 'w').close() 
            else:
                pass
        else:
            open(path + filename, 'w').close()
    else:
        os.mkdir(path)
        open(path + filename, 'w').close()

def is_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

def get_results_dir(dataset_name):
    return get_module_dir('results/', dataset_name)

def get_data_dir(dataset_name):
    return get_module_dir('data/', dataset_name)

def get_module_dir(module, dataset_name):
    is_dir(BB_DIR + module)
    module_dir = os.path.abspath(os.path.join(
        BB_DIR, 
        module,
        dataset_name,
    ))
    is_dir(module_dir)
    return module_dir + '/'

def call_bash(command, message=None):
    try:
        output = subprocess.check_output(
            command, 
            shell=True, 
            stderr=subprocess.STDOUT,
        )
        if message is not None:
            print(message) 
        return output
    except subprocess.CalledProcessError as e:
        print(e.output)
        print("Exiting program...")
        sys.exit()





