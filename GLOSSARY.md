# Glossary

## Main

* **version**: The version number assigned to the current model being trained on. \
Note: CHECK VERSION NUMBER! Users should get into the habit of checking which version number is being set in order to avoid overwriting old models (unless that is the intended results).

* **vid_dir**: Absolute path to where video files are stored to. \
Note: do not set path to within the project's 'data' or 'results' directories.

* **dataset**: Video set and accompanying data files which the user wishes to download. \
Supported datasets include: kinetics400, kinetics600, kinetics700, kinetics700_2020, HACS, actnet100, and actnet200.

* **use_records**: Will switch the model over to use TFRecords. \
Note: the TFRecords dataset will be much larger than its Tensorflow Dataset counterpart but will lead to faster training times. If operational cost is more important than memory cost, set this param to True. \
Note: unless the user has a reason to keep the videos after creating the TFRecords, it is recommended to set delete_videos to True in order to avoid massive use of memory space.

* **use_cloud**: Set to True if user wished to retrieve data from a remote location. \
Note: the code has been set up to work with Google Cloud Storage and most likely will not work with any other service.

* **batch_per_replica**: Total number of samples passed to each replica. \
Do not set this number to a global value since this value will be passed to each device available.

## Ingestion

* **cookies**: Absolute path to cookies file, which will be passed to youtube-dl during downloading. \
Passing a cookies file is necessary since many websites will require you to login to authenticate yourself. In order to do so, please install the extension, [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid?hl=en). After that, simply go to the website you wish to download videos from, login, click on the extension, and 'export'. The cookies file should be found in your 'Downloads' directory.\
Note: do not set path to within the project's 'data' or 'results' directories.

* **conda_path**: Absolute path to conda package (if running a conda virtual environment). This is optional as some users may wish to use Pipenv or no virtual environment at all (not recommended). \
The reason this is a parameter is because the bash scripts do not recognize when a conda environment has been activated, and hence the scripts will fail. So we must manually pass the package directory to the scripts.

* **conda_env**: Name of your Conda virtual environment. This is optional as some users may wish to use Pipenv or no virtual environment at all (not recommended). \
The reason this is a parameter is because the bash scripts do not recognize when a conda environment has been activated, and hence the scripts will fail. So we must manually pass the environment's name to the scripts.

* **num_jobs**: The number of simultaneous (using GNU Parallel) videos to download. A good value to use is one equal to the total number of CPU cores on your machine. 

* **toy_set**: Set to True if user wishes to work a smaller subset to experiment with. \
Note: this is for experimentation only, as any results gathered from this toy set will not be accurate.

* **toy_samples**: Total number of videos to subset for the toy dataset.

* **download_batch**: Batch size of total videos to download on each iteration. This batch size will be evenly distributed to all the workers (i.e. num_jobs). \
Note: Do not set this value too high since HTTP Errors may occur that will terminate the downloading loop. At that point, all videos that were successfully download with that batch will be discarded in favor or rerunning that iteration.

* **download_fps**: This is the frame rate at which FFMPEG will download each video at.

* **time_interval**: Fixed time length (starting from a defined start time) that all videos in the set will be downloaded at (in seconds).

* **shorter_edge**: The maximum length of a frame's shorter side that FFMPEG will download all videos at.

* **use_sampler**: Set to True if user wishes to sample only a portion of the video right away. \
Some datasets do not come with predefined stating points. For these cases, the sampler will auto generate the starting points for you.

* **max_duration**: This is the max length (in seconds) of all the combined samples belonging to a single video. \
Example: If we wish to collect 10 samples from a single video and set max_duration to 30 seconds, then each sample will be 3 seconds long. \
Note: If the video is shorter than the max_duration, this param will be ignored.

* **num_samples**: Total number of samples to collect from each video.

* **sampling**: Sampling technique to use. Can be one of 'random' or 'uniform' sampling.

* **sample_duration**: Duration of each sampled clip. While max_duration will limit the combined times of all samples, sample_duration will further limit the total duration of a single sample. \
Example (following example from max_duration): with each sample currently sitting at 3 seconds long, sample_duration could further trim its value to say 1 second. \
The reason for both these params is that max_duration is more of a grouping of time intervals, while sample_duration hones in on a specific value at each grouping.

* **videos_per_record**: Total number of videos to be placed in each training TFRecord. \
This is the total number of videos per record, not total number of examples per record. \
Example: If videos_per_record is set to 125 and train_clips_per_vid is set to 4, then there will be 125*4=500 total examples in each record.

* **videos_per_test_record**: Total number of videos to be placed in each validation/testing TFRecord. \
This is the total number of videos per test record, not total number of examples per test record. \
Example: If videos_per_test_record is set to 50 and test_clips_per_vid is set to 10, then there will be 50*10=500 total examples in each record. \
Note: due to 'validation' and 'test' sharing the same tfrecords, the total number of examples in each validation record will be determined by the test_clips_per_vid value.

* **fps**: Frame rate at which each video will be sampled from during TFRecord creation and/or preprocessing. \
Note: fps should be less than or equal to download_fps.

* **frames_per_clip**: Total number of frames per clip. \
Supported values include: 1, 2, 4, 8, or any factor of 8.

* **train_clips_per_vid**: Total number of training clips to sample from each video.

* **val_clips_per_vid**: Total number of validation clips to sample from each video. \
Note: val_clips_per_vid should be less than or equal to test_clips_per_vid.

* **test_clips_per_vid**: Total number of testing clips to sample from each video. \
Note: test_clips_per_vid should be greater than or equal to val_clips_per_vid.

* **delete_videos**: Set to True if user wished to delete videos after saving files to TFRecords. \
Note: unless the user has a reason to keep the videos after creating the TFRecords, it is recommended to set delete_videos to True in order to avoid massive use of memory space.

* **interleave_num**: Total number of records to simultaneously interleave. \
As a visual example: for a simplest case, allow your hands to represent two records and your fingers to represent examples from each each record. If you now interlock your hands together, this will represent the interleaving performed by this operation. \
This option is necessary for TFRecords since you cannot shuffle binary data. \
For further information, check the [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave) documentation on this operation.

* **interleave_block**: Total number of consecutive elements that are interwoven from each record at one time. \
This option just allows for blocks of examples from the same record to be placed next to one another when interwoven with examples (also in a block) from another record. \
For further information, check the [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave) documentation on this operation.

* **gcs_data**: Remote path where TFRecords are stored. \
Note: the code has been set up to work with Google Cloud Storage and most likely will not work with any other service. \
Note: The directory defined by this path should contain the 'train' and 'validate' directories. But they should be omitted from the param's string value.

## Preprocess

* **spatial_size**: Final length used to crop each frame to. \
Supported values include: 112, 128, 224, and 256.

* **shuffle_size**: Sample size to shuffle and place into buffer during preprocessing. \
A good starting value may be equal to the total number of examples in 1 or 2 records. This value is low in order to avoid out-of-memory issues. But ultimately, the optimal value (higher the better) will be dependent on the user's machine.

* **mean**: Mean values calculated (from train dataset) for each RGB channel.

* **std**: STD values calculated (from train dataset) for each RGB channel.

* **data_format**: The ordering of the dimensions. \
Supported values include: 'channels_last' (NTHWC) or 'channels_first' (NCTHW).

## Models

* **arch**: Model architecture to be used. \
Supported architectures include: 'ip-csn' and 'ir-csn'.

* **depth**: Total number of convolutional layers used by the model. \
Supported values include: 26, 50, 101, and 152.

* **regularization**: Regularization technique to be used. \
Supported techniques include: 'weight_decay' and 'l2'. \
The difference between these two techniques is that weight_decay achieves regularization by decaying the resultant variable from the optimizer's update step, meanwhile L2 achieves regularization by adding a regularization loss to the total loss. \
For further information, check ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) and the documentation on the [SGDW](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SGDW) optimizer.

* **weight_decay**: Regularization decay factor to use on the resultant variable from the optimizer's update step.

* **initializer**: Technique to initialize the model weights. \
Recommended initializers include: 'glorot_normal', 'glorot_uniform', 'he_normal', and 'he_uniform'. \
If the user wishes to use a different initializer not in this list, please check with the [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/initializers) documentation first to determine if your requested initializer is supported. If not, you may receive an error at runtime.

* **use_bias**: Set to True if user wishes to include a bias term with each layer.

* **bn_momentum**: Defines the momentum for the moving average used in each batch norm layer.

* **epsilon**: Value added to each batch norm's variance to avoid division by zero.

* **dropout_rate**: The percentage of units to drop at a specific layer. \
Recommended value is 0.2 (if used). Otherwise, default value of 0 will effectively lead to no dropout used.

## Train / Test

* **is_eager**: Set to True if user wishes to run tf.functions in eager mode. This can be set to True when debugging the code, but should be set to False when actually training the model.

* **strategy**: Tensorflow Strategy to use during training. \
Supported strategies include: 'default', 'mirrored', and 'tpu'. \
Use default when using only a single GPU, use mirrored when using multiple GPUs, and use tpu when using one or multiple TPUs.

* **tpu_address**:  Set this param either to the TPU's name or the TPU's gRPC address.

* **gcs_results**: Remote path where SaveModels are stored. \
Note: the code has been set up to work with Google Cloud Storage and most likely will not work with any other service. \
Note: do not include the version number at the end of the path name.

* **mixed_precision**: Set to True if user wishes to use a mix of 16 and 32 bit floating-point types. This should greatly reduce the memory footprint and minimal loss to accuracy. \
Note: this param does not currently work when used with GPUs. Only available for TPUs.

* **num_epochs**: Total number of epochs to train the model for.

* **epoch_size**: Total number of clips to use per epoch. If set, will limit the total number of clips seen during each epoch instead of exhausting the whole dataset.

* **steps_per_execution**: Total number of batches to simultaneously push through each tf.function at one time. \
This a brand new feature so there is not much documentation available yet (although faster training times are claimed). So the optimal value chosen will have to be up to the user. If set to 1, this model will run as normal (one step at a time). \
Note: the only limitation on this param is that 'track_every' must be a multiple of 'steps_per_execution'.

* **use_warmup**: Set to True if user wishes to use the model warming-up technique.

* **warmup_factor**: Starting decay factor used when applying the model warmup technique.

* **use_half_cosine**: Set to True if user wishes to use a half cosine decay (post model warmup).

* **decay_epoch**: Determines how often (i.e. every nth epoch) to apply a decay rate to the learning rate.

* **decay_rate**: Factor to decay the learning rate by.

* **lr_per_replica**: Initial learning rate used in calculating the training step size. \
Note: this value will be passed to all available devices (GPU or TPU).

* **momentum**: Used by optimizer to accelerate gradient descent and dampen oscillations.

* **nesterov**: Set to True if user wishes to apply Nesterov momentum to the optimizer.

* **track_every**: Determines how often to save a set of values ('best', 'step', 'epoch') needed in case the model is interrupted and needs to be restored. \
Note: 'track_every' must be a multiple of 'steps_per_execution'.

* **max_k**: Total number of top predictions to use in calculating the accuracy metric.

## Hidden

* **switch**: Globally switches the whole codebase from using the base config file to a specific frozen config file (i.e. saved copy of a versioned model). \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model.

* **num_replicas**: Defines the total number of devices (CPU, GPU, or TPU) available. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model.

* **train_size**: Defines the total number of training examples found in all TFRecords. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved using DataGenerator's _set_split_size method.

* **validate_size**: Defines the total number of validation/testing examples found in all TFRecords. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved using DataGenerator's _set_split_size method.

* **train_unusable**: Defines total number of examples that break the model. These are from videos in which the provided start times were wrong or the video has possibly been shorted since the dataset was first created. In either case, the video is filtered out. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved by looping through all the TFRecords or using tf.Dataset's reduce function. Fair warning: the operation takes a while to complete.

* **validate_unusable**: Defines total number of examples that break the model. These are from videos in which the provided start times were wrong or the video has possibly been shorted since the dataset was first created. In either case, the video is filtered out. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved by looping through all the TFRecords or using tf.Dataset's reduce function. Fair warning: the operation takes a while to complete.

* **num_labels**: Defines the total number of labels available for a specific dataset. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved using DataGenerator's _set_labels method.

* **loop_it**: keeps track of downloading iteration in case the downloading loop is interrupted. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model.


