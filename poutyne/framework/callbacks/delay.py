
from .callbacks import Callback, CallbackList

class DelayCallback(Callback):
    """
    Delays one or many callbacks for a certain number of epochs or number of
    batches. If both ``epoch_delay`` and ``batch_delay`` are provided, the
    longer one has precedence.

    Args:
        callbacks (Callback, list of Callback): A callback or a list of
            callbacks to delay.
        epoch_delay (int, optional): Number of epochs to delay.
        batch_delay (int, optional): Number of batches to delay. The number of
            batches can span many epochs. When the batch delay expires (i.e.
            there are more than `batch_delay` done), the ``on_epoch_begin``
            method is called on the callback(s) before the ``on_batch_begin``
            method.
    """
    def __init__(self, callbacks, *, epoch_delay=None, batch_delay=None):
        super().__init__()
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        elif isinstance(callbacks, list):
            self.callbacks = CallbackList(callbacks)
        else:
            self.callbacks = CallbackList([callbacks])

        self.epoch_delay = epoch_delay if epoch_delay else 0
        self.batch_delay = batch_delay if batch_delay else 0

    def set_params(self, params):
        self.callbacks.set_params(params)

    def set_model(self, model):
        self.callbacks.set_model(model)

    def on_epoch_begin(self, epoch, logs):
        self.current_epoch = epoch
        if self.has_delay_passed():
            self.has_on_epoch_begin_been_called = True
            self.callbacks.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        if self.has_delay_passed():
            self.callbacks.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs):
        self.batch_counter += 1
        if self.has_delay_passed():
            if not self.has_on_epoch_begin_been_called:
                self.has_on_epoch_begin_been_called = True
                self.callbacks.on_epoch_begin(self.current_epoch, logs)
            self.callbacks.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs):
        if self.has_delay_passed():
            self.callbacks.on_batch_end(batch, logs)

    def on_backward_end(self, batch):
        if self.has_delay_passed():
            self.callbacks.on_backward_end(batch)

    def on_train_begin(self, logs):
        self.current_epoch = 0
        self.batch_counter = 0
        self.has_on_epoch_begin_been_called = False
        self.callbacks.on_train_begin(logs)

    def on_train_end(self, logs):
        self.callbacks.on_train_end(logs)

    def has_delay_passed(self):
        return self.current_epoch > self.epoch_delay and \
                self.batch_counter > self.batch_delay
