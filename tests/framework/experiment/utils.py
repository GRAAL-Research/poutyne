from poutyne import Callback


class ConstantMetric:
    __name__ = 'const'

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, *args, **kwds):
        return self.value


class ConstantMetricCallback(Callback):
    def __init__(self, values, constant_metric):
        super().__init__()
        self.values = values
        self.constant_metric = constant_metric

    def on_epoch_begin(self, epoch_number, logs):
        self.constant_metric.value = (
            self.values[epoch_number - 1] if epoch_number < len(self.values) else self.values[-1]
        )
