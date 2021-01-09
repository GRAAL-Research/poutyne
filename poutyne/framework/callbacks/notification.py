from typing import Dict, Union

from notif import Notificator

from . import Callback


class NotificationCallback(Callback):
    """
    Send a notification to a channel at the beginning/ending of the training/testing and at a constant frequency
    (`alert_frequency`) during the training.

    Args:
        notificator (~notif.Notificator): The notification channel to send the message. It's a Notificator class define
            by the `notif <https://notificationdoc.ca/index.html>`_ package.
        alert_frequency (int): The frequency (in epoch), during training, to send update. By default 1.
        experiment_name (Union[str, None]): The name of the experiment to add into the message. By default None.
    """

    def __init__(self,
                 notificator: Notificator,
                 alert_frequency: int = 1,
                 experiment_name: Union[None, str] = None) -> None:
        super().__init__()
        self.notificator = notificator
        self.alert_frequency = alert_frequency
        self.experiment_name_msg = f" for {experiment_name}" if experiment_name is not None else ""

    def on_train_begin(self, logs: Dict) -> None:
        """
        Send the message 'Start of the training< for the experiment experiment_name>' to the channel.
        """
        empty_message = ""
        self.notificator.send_notification(empty_message, subject=f"Start of the training{self.experiment_name_msg}.")

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        """
        Send the message 'Epoch is done< for the experiment experiment_name>' plus the logs metrics (one per line)
        to the channel.
        """

        if epoch_number % self.alert_frequency == 0:
            message = f"Here the epoch metrics: \n{self._format_logs(logs)}"
            self.notificator.send_notification(message,
                                               subject=f"Epoch {epoch_number} is done{self.experiment_name_msg}.")

    def on_train_end(self, logs: Dict) -> None:
        """
        Send the message 'End of the training< for the experiment experiment_name>' to the channel.
        """

        empty_message = ""
        self.notificator.send_notification(empty_message, subject=f"End of the training{self.experiment_name_msg}.")

    def on_test_begin(self, logs: Dict) -> None:
        """
        Send the message 'Start of the testing< for the experiment experiment_name>' to the channel.
        """

        empty_message = ""
        self.notificator.send_notification(empty_message, subject=f"Start of the testing{self.experiment_name_msg}.")

    def on_test_end(self, logs: Dict) -> None:
        """
        Send the message 'End of the testing< for the experiment experiment_name>' to the channel.
        """

        message = f"Here the epoch metrics: \n{self._format_logs(logs)}"
        self.notificator.send_notification(message, subject=f"End of the testing{self.experiment_name_msg}.")

    @staticmethod
    def _format_logs(logs: Dict) -> str:
        return " ".join([f"{key}: {value}\n" for key, value in logs.items()])
