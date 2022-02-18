"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

from typing import Dict

from .callbacks import Callback


class LambdaCallback(Callback):
    """
    Provides an interface to easily define a callback from lambdas or functions.

    Args:
        kwargs: The arguments of this class are keyword arguments with the same names as the methods in the
            :class:`~poutyne.Callback` class. The values are lambdas or functions taking the same arguments as the
            corresponding methods in :class:`~poutyne.Callback`.

    See:
        :class:`~poutyne.Callback`

    Example:
        .. code-block:: python

            from poutyne import LambdaCallback
            callbacks = [LambdaCallback(
                on_epoch_end=lambda epoch_number, logs: print(f"Epoch {epoch_number} end"),
                on_train_end=lambda logs: print("Training ended")
            )]
            model.fit(...., callbacks=callbacks)
    """

    def __init__(
        self,
        *,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_train_batch_begin=None,
        on_train_batch_end=None,
        on_valid_batch_begin=None,
        on_valid_batch_end=None,
        on_test_batch_begin=None,
        on_test_batch_end=None,
        on_predict_batch_begin=None,
        on_predict_batch_end=None,
        on_train_begin=None,
        on_train_end=None,
        on_valid_begin=None,
        on_valid_end=None,
        on_test_begin=None,
        on_test_end=None,
        on_predict_begin=None,
        on_predict_end=None,
        on_backward_end=None
    ):
        # pylint: disable=too-many-locals
        super().__init__()
        self._on_epoch_begin = self._set_lambda_for_none(on_epoch_begin)
        self._on_epoch_end = self._set_lambda_for_none(on_epoch_end)
        self._on_train_batch_begin = self._set_lambda_for_none(on_train_batch_begin)
        self._on_train_batch_end = self._set_lambda_for_none(on_train_batch_end)
        self._on_valid_batch_begin = self._set_lambda_for_none(on_valid_batch_begin)
        self._on_valid_batch_end = self._set_lambda_for_none(on_valid_batch_end)
        self._on_test_batch_begin = self._set_lambda_for_none(on_test_batch_begin)
        self._on_test_batch_end = self._set_lambda_for_none(on_test_batch_end)
        self._on_predict_batch_begin = self._set_lambda_for_none(on_predict_batch_begin)
        self._on_predict_batch_end = self._set_lambda_for_none(on_predict_batch_end)
        self._on_train_begin = self._set_lambda_for_none(on_train_begin)
        self._on_train_end = self._set_lambda_for_none(on_train_end)
        self._on_valid_begin = self._set_lambda_for_none(on_valid_begin)
        self._on_valid_end = self._set_lambda_for_none(on_valid_end)
        self._on_test_begin = self._set_lambda_for_none(on_test_begin)
        self._on_test_end = self._set_lambda_for_none(on_test_end)
        self._on_predict_begin = self._set_lambda_for_none(on_predict_begin)
        self._on_predict_end = self._set_lambda_for_none(on_predict_end)
        self._on_backward_end = self._set_lambda_for_none(on_backward_end)

    def _set_lambda_for_none(self, value):
        return value if value is not None else lambda *args: None

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        self._on_epoch_begin(epoch_number, logs)

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        self._on_epoch_end(epoch_number, logs)

    def on_train_batch_begin(self, batch_number: int, logs: Dict):
        self._on_train_batch_begin(batch_number, logs)

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        self._on_train_batch_end(batch_number, logs)

    def on_valid_batch_begin(self, batch_number: int, logs: Dict):
        self._on_valid_batch_begin(batch_number, logs)

    def on_valid_batch_end(self, batch_number: int, logs: Dict):
        self._on_valid_batch_end(batch_number, logs)

    def on_test_batch_begin(self, batch_number: int, logs: Dict):
        self._on_test_batch_begin(batch_number, logs)

    def on_test_batch_end(self, batch_number: int, logs: Dict):
        self._on_test_batch_end(batch_number, logs)

    def on_predict_batch_begin(self, batch_number: int, logs: Dict):
        self._on_predict_batch_begin(batch_number, logs)

    def on_predict_batch_end(self, batch_number: int, logs: Dict):
        self._on_predict_batch_end(batch_number, logs)

    def on_train_begin(self, logs: Dict):
        self._on_train_begin(logs)

    def on_train_end(self, logs: Dict):
        self._on_train_end(logs)

    def on_valid_begin(self, logs: Dict):
        self._on_valid_begin(logs)

    def on_valid_end(self, logs: Dict):
        self._on_valid_end(logs)

    def on_test_begin(self, logs: Dict):
        self._on_test_begin(logs)

    def on_test_end(self, logs: Dict):
        self._on_test_end(logs)

    def on_predict_begin(self, logs: Dict):
        self._on_predict_begin(logs)

    def on_predict_end(self, logs: Dict):
        self._on_predict_end(logs)

    def on_backward_end(self, batch_number: int):
        self._on_backward_end(batch_number)
