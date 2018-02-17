import numpy as np

from .callbacks import Callback


class TerminateOnNaN(Callback):
    def __init__(self):
        super(TerminateOnNaN, self).__init__()

    def on_batch_end(self, batch, logs):
        loss = logs['loss']
        if np.isnan(loss) or np.isinf(loss):
            print('Batch %d: Invalid loss, terminating training' % (batch))
            self.model.stop_training = True
