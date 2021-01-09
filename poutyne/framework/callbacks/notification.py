from typing import Dict, Union

from notif import Notificator

from . import Callback


class NotificationCallback(Callback):

    def __init__(self, notificator: Notificator, alert_frequency: int = 1, experiment_name: Union[None, str] = None):
        super().__init__()
        self.notificator = notificator
        self.alert_frequency = alert_frequency
        self.experiment_name_msg = f" for {experiment_name}" if experiment_name is not None else ""

    def on_train_begin(self, logs: Dict):
        empty_message = ""
        self.notificator.send_notification(empty_message, subject=f"Start of the training{self.experiment_name_msg}.")

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        if epoch_number % self.alert_frequency == 0:
            message = f"Here the epoch metrics: \n{self._format_logs(logs)}"
            self.notificator.send_notification(message, subject=f"Epoch {epoch_number} is done.")

    def on_train_end(self, logs: Dict):
        empty_message = ""
        self.notificator.send_notification(empty_message, subject=f"End of the training{self.experiment_name_msg}.")

    @staticmethod
    def _format_logs(logs: Dict) -> str:
        return " ".join([f"{key}: {value}\n" for key, value in logs.items()])
