import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

from utils import get_frozen_params, get_params, get_results_dir, is_file


PARAMS = get_params('Test')
VERSION = PARAMS['version']
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Test',
        version=VERSION,
    )


def csvlogger():
    logs_dir = get_results_dir(PARAMS['dataset']) + 'test-logs/'
    is_file(
        logs_dir, 
        filename=f'test_logs_v{VERSION}.csv', 
    )
    return CSVTestLogger(
        logs_dir + f'test_logs_v{VERSION}.csv',
    )

class CSVTestLogger(CSVLogger):
    def __init__(self, filename):
        super(CSVTestLogger, self).__init__(filename=filename)

    def on_test_begin(self, logs=None):
        super(CSVTestLogger, self).on_train_begin(logs=logs)

    def on_test_end(self, logs=None):
        super(CSVTestLogger, self).on_epoch_end(0, logs=logs)
        super(CSVTestLogger, self).on_train_end(logs=logs)