from typing import Dict

import numpy as np

from .callbacks import Callback


class TerminateOnNaN(Callback):
    """
    Stops the training when the loss is either `NaN` or `inf`.
    """

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        loss = logs['loss']
        if np.isnan(loss) or np.isinf(loss):
            print('Batch %d: Invalid loss, terminating training' % (batch_number))
            self.model.stop_training = True
