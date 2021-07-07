# BB-Action

BB-Action is a Tensorflow 2 training pipeline for video classification. The aim of this codebase is to set up a foundation on which future projects might be able to build upon. Supported models currently include:

  * **Channel-Separated Convolutional Networks** [[1]](https://arxiv.org/pdf/1904.02811.pdf)
                                                                                
**Note #1: This repository has only been tested on Ubuntu 18.04 and Debian (Sid).**

**Note #2: This project is dependent on BB-data, which is a separate project.**

## Installation

  1. Clone the repositories:
  ```
  git clone https://github.com/salinasJJ/BBaction.git 
  git clone https://github.com/salinasJJ/BBdata.git
  ```
  2. Create a virtual environment (using Pipenv or Conda for example).

  3. Install the projects onto your system:
  ```
  pip install -e BBaction
  pip install -e BBdata
  ```
  4. Using apt, install the following:         
  ```
  apt install parallel
  apt install ffmpeg
  apt install aria2c
  ```                                                                   
  5. Install dependencies:
  ```
  pip install -r BBaction/bbaction/requirements.txt
  ```
  6. Make scripts executable: 
  ```
  chmod +x BBdata/bbdata/download/scripts/data.sh         
  chmod +x BBdata/bbdata/download/scripts/setup.sh         
  chmod +x BBdata/bbdata/download/scripts/videos.sh         
  ```
    
## Params    

This project is setup around a config file which contains numerous adjustable parameters. Any changes to how the project runs must be done here by updating the params. There are 3 main commands to update the params:
  1. The 'reset' command will reset all params back to their default values:
  ```
  python BBaction/bbaction/main.py reset
  ```      
  2. The 'update' command will update all requested params to new values. For example:                 
  ```
  python BBaction/bbaction/main.py update \
    --version 0 \
    --dataset kinetics400 \
    --batch_per_replica 8 \
    --use_records True \
    --num_epochs 45 \
    --strategy mirrored \
  ```
  3. The 'force' command is a special command that will update a set of model-defined hidden params. The command is there for a specific use case (i.e. resetting hidden params after an accidental updating) but in general, users of this repository should never have to use this command.
  ```
  python BBaction/bbaction/main.py force \
    --train_size 835516 \
    --validate_size 171890 \
    --train_unusable 40 \
    --validate_unusable 10 \
    --num_labels 400 \
  ```

**Note #1: The 'reset' command will clear out all user-defined values as well as those of the hidden params. Without these pre-defined params, the model will fail to work properly, if at all. Please use this command carefully.**

**Note #2: CHECK VERSION NUMBER! Users should get into the practice of always checking which version number is being set in order to avoid overwriting old models.**

**Note #3: Do not set the paths to 'cookies' or 'vid_dir' within the data directory. Best to place them in directories outside of the project.**

There are many params, some of which are interconnected to one another, and some which have limitations. Please see [Glossary](GLOSSARY.md) for a full breakdown of all these params.

That was the hard part. From here on out, the commands to create the datasets, train, and test are simple one liners.

## Datasets

In order to create the datasets, we can make use of the 'ingest' command. This command contains two options:
  1. setup: retrieves required data files and prepares the data for downloading.
  2. generate: downloads the video set and generates Tensorflow datasets.

To setup and start downloading, call:
```
python BBaction/bbaction/main.py ingest --setup --generate
```

**Note: The 'setup' option will clear everything in the data directory. So, if downloading is interrupted, make sure to only use 'generate' to restart downloading.**
```
python BBaction/bbaction/main.py ingest --generate
```

## Training

To start training on a brand new model, call:
```
python BBaction/bbaction/main.py train
```
If training is interupted for any reason, you can easily restart from where you left off:
```
python BBaction/bbaction/main.py train --restore
```

## Testing

To evaluate your trained model:
```
python BBaction/bbaction/main.py test
```

## Contribute

Contributions from the community are welcomed.

## License

BB-Action is licensed under MIT.

## References

  1. D. Tran, H. Wang, L. Torresani, M. Feiszli, **Video Classification with Channel-Separated Convolutional Networks**, arXiv:1904.02811, 2019.