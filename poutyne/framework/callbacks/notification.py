from abc import ABC, abstractmethod
from typing import Dict, Union

from . import Callback


class Notificator(ABC):
    """
    The interface of the Notificator. It must at least implement a `send_notification` method.
    The interface is similar to the `notif <https://notificationdoc.ca/index.html>`_ package.
    """

    @abstractmethod
    def send_notification(self, message: str, subject: Union[str, None] = None) -> None:
        """
        Abstract method to send a notification.

        Args:

            message (str): The message to send as a notification message through the notificator.
            subject (str): The subject of the notification. If None, the default message is used. By default, None.
                Also, we recommend formatting the subject for better readability, e.g. using bolding it using Markdown
                and appending with a new line '*Title*\n'.
        """


class NotificationCallback(Callback):
    """
    Send a notification to a channel at the beginning/ending of the training/testing and at a constant frequency
    (`alert_frequency`) during the training.

    Args:
        notificator (~poutyne.Notificator): The notification channel to send the message.
            The expected interface need to implement a `send_notification` method to send the message. You can see the
            `notif <https://notificationdoc.ca/index.html>`_ package which implements some Notificator respecting the
            interface.
        alert_frequency (int): The frequency (in epoch), during training, to send an update. By default, 1.
        experiment_name (Union[str, None]): The name of the experiment to add to the message. By default, None.
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
