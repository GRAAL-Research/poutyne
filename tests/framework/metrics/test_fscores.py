"""
The source code of this file was copied from the AllenNLP project, and has been modified. All modifications
made from the original source code are under the LGPLv3 license.


Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information on the Poutyne and AllenNLP repository.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.


Copyright 2019 AllenNLP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=protected-access
from unittest import TestCase

import numpy
import torch
import torch.nn as nn

from poutyne import FBeta, F1, BinaryF1, Model


class FBetaTest(TestCase):
    def setUp(self):
        # [0, 1, 1, 1, 3, 1]
        self.predictions = torch.Tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ]
        )
        self.targets = torch.Tensor([0, 4, 1, 0, 3, 0])

        # detailed target state
        self.pred_sum = [1, 4, 0, 1, 0]
        self.true_sum = [3, 1, 0, 1, 1]
        self.true_positive_sum = [1, 1, 0, 1, 0]
        self.true_negative_sum = [3, 2, 6, 5, 5]
        self.total_sum = [6, 6, 6, 6, 6]

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00]
        desired_recalls = [0.33, 1.00, 0.00, 1.00, 0.00]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0 for p, r in zip(desired_precisions, desired_recalls)
        ]
        self.desired_precisions = desired_precisions
        self.desired_recalls = desired_recalls
        self.desired_fscores = desired_fscores

    def test_config_errors(self):
        # Bad beta
        self.assertRaises(ValueError, FBeta, beta=0.0)

        # Bad average option
        self.assertRaises(ValueError, FBeta, average='mega')

        # F1 classes with beta different than 1.
        self.assertRaises(ValueError, F1, beta=2.0)
        self.assertRaises(ValueError, BinaryF1, beta=2.0)

        # Precision and recall with beta different than 1.
        self.assertWarns(UserWarning, FBeta, metric='precision', beta=2.0)
        self.assertWarns(UserWarning, FBeta, metric='recall', beta=2.0)

    def test_runtime_errors(self):
        fbeta = FBeta()
        # Metric was never called.
        self.assertRaises(RuntimeError, fbeta.compute)

    def test_fbeta_multiclass_state(self):
        fbeta = FBeta()
        fbeta.update(self.predictions, self.targets)

        # check state
        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), self.pred_sum)
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), self.true_sum)
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        numpy.testing.assert_almost_equal(fbeta._total_sum.tolist(), self.total_sum)

    def test_fbeta_multiclass_with_mask(self):
        mask = torch.Tensor([1, 1, 1, 1, 1, 0])

        fbeta = FBeta()
        fbeta.update(self.predictions, (self.targets, mask))

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 3, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [2, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), [1, 1, 0, 1, 0])

    def test_fbeta_multiclass_with_ignore_index(self):
        targets = self.targets.clone()
        targets[-1] = -100
        fbeta = FBeta()
        fbeta.update(self.predictions, targets)

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 3, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [2, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), [1, 1, 0, 1, 0])

    def test_fbeta_multiclass_with_mask_and_ignore_index(self):
        targets = self.targets.clone()
        targets[-1] = -100
        mask = torch.Tensor([1, 1, 1, 1, 0, 1])

        fbeta = FBeta()
        fbeta.update(self.predictions, (targets, mask))

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 3, 0, 0, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [2, 1, 0, 0, 1])
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), [1, 1, 0, 0, 0])

    def test_fbeta_multiclass_macro_average_metric(self):
        precision = self._compute(metric='precision', average='macro')
        recall = self._compute(metric='recall', average='macro')
        fscore = self._compute(metric='fscore', average='macro')

        macro_precision = numpy.mean(self.desired_precisions)
        macro_recall = numpy.mean(self.desired_recalls)
        macro_fscore = numpy.mean(self.desired_fscores)

        # check type
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fscore, float)

        # check value
        numpy.testing.assert_almost_equal(precision, macro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recall, macro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscore, macro_fscore, decimal=2)

    def test_fbeta_multiclass_micro_average_metric(self):
        precision = self._compute(metric='precision', average='micro')
        recall = self._compute(metric='recall', average='micro')
        fscore = self._compute(metric='fscore', average='micro')

        true_positives = [1, 1, 0, 1, 0]
        false_positives = [0, 3, 0, 0, 0]
        false_negatives = [2, 0, 0, 0, 1]
        mean_true_positive = numpy.mean(true_positives)
        mean_false_positive = numpy.mean(false_positives)
        mean_false_negative = numpy.mean(false_negatives)

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        # check value
        numpy.testing.assert_almost_equal(precision, micro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recall, micro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscore, micro_fscore, decimal=2)

    def test_fbeta_multiclass_with_explicit_label(self):
        # same prediction but with and explicit label
        label = 3
        precision = self._compute(metric='precision', average=label)
        recall = self._compute(metric='recall', average=label)
        fscore = self._compute(metric='fscore', average=label)

        desired_precision = self.desired_precisions[label]
        desired_recall = self.desired_recalls[label]
        desired_fscore = self.desired_fscores[label]
        # check value
        numpy.testing.assert_almost_equal(precision, desired_precision, decimal=2)
        numpy.testing.assert_almost_equal(recall, desired_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscore, desired_fscore, decimal=2)

    def test_fbeta_multiclass_macro_average_metric_multireturn(self):
        fbeta = FBeta(average='macro')
        fbeta.update(self.predictions, self.targets)
        fscore, precision, recall = fbeta.compute()

        macro_precision = numpy.mean(self.desired_precisions)
        macro_recall = numpy.mean(self.desired_recalls)
        macro_fscore = numpy.mean(self.desired_fscores)

        # check type
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fscore, float)

        # check value
        numpy.testing.assert_almost_equal(precision, macro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recall, macro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscore, macro_fscore, decimal=2)

    def test_fbeta_handles_batch_size_of_one(self):
        predictions = torch.Tensor([[0.2862, 0.3479, 0.1627, 0.2033]])
        targets = torch.Tensor([1])
        mask = torch.Tensor([1])

        fbeta = FBeta()
        fbeta.update(predictions, (targets, mask))

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [0.0, 1.0, 0.0, 0.0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [0.0, 1.0, 0.0, 0.0])
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), [0.0, 1.0, 0.0, 0.0])
        numpy.testing.assert_almost_equal(fbeta._total_sum.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_fbeta_with_return_batch_value(self):
        targets = self.targets.clone()
        targets[-1] = -100
        mask = torch.Tensor([1, 1, 1, 1, 0, 1])

        fbeta = FBeta()
        batch_value = fbeta(self.predictions, (targets, mask))
        epoch_value = fbeta.compute()

        self.assertEqual(batch_value, epoch_value)

    def _compute(self, *args, **kwargs):
        fbeta = FBeta(*args, **kwargs)
        fbeta.update(self.predictions, self.targets)
        return fbeta.compute()

    def test_names(self):
        fbeta = FBeta(average='macro')
        self.assertEqual(['fscore_macro', 'precision_macro', 'recall_macro'], fbeta.__name__)
        fbeta = FBeta(average='micro')
        self.assertEqual(['fscore_micro', 'precision_micro', 'recall_micro'], fbeta.__name__)
        fbeta = FBeta(average='micro', names=['f', 'p', 'r'])
        self.assertEqual(['f', 'p', 'r'], fbeta.__name__)
        fbeta = FBeta(average=0)
        self.assertEqual(['fscore_0', 'precision_0', 'recall_0'], fbeta.__name__)
        fbeta = FBeta(metric='fscore', average='macro')
        self.assertEqual('fscore_macro', fbeta.__name__)
        fbeta = FBeta(metric='fscore', average='micro')
        self.assertEqual('fscore_micro', fbeta.__name__)
        fbeta = FBeta(metric='fscore', average=0)
        self.assertEqual('fscore_0', fbeta.__name__)
        fbeta = FBeta(metric='precision', average='macro')
        self.assertEqual('precision_macro', fbeta.__name__)
        fbeta = FBeta(metric='precision', average='micro')
        self.assertEqual('precision_micro', fbeta.__name__)
        fbeta = FBeta(metric='precision', average=0)
        self.assertEqual('precision_0', fbeta.__name__)
        fbeta = FBeta(metric='recall', average='macro')
        self.assertEqual('recall_macro', fbeta.__name__)
        fbeta = FBeta(metric='recall', average='micro')
        self.assertEqual('recall_micro', fbeta.__name__)
        fbeta = FBeta(metric='recall', average=0)
        self.assertEqual('recall_0', fbeta.__name__)
        fbeta = FBeta(metric='fscore', average='macro', names='f')
        self.assertEqual('f', fbeta.__name__)
        fbeta = FBeta(average='macro', names=['f', "p", "r"])
        self.assertEqual(["f", "p", "r"], fbeta.__name__)
        fbeta = FBeta(average='binary')
        self.assertEqual(['fscore_binary_1', 'precision_binary_1', 'recall_binary_1'], fbeta.__name__)
        fbeta = FBeta(average='binary', pos_label=0)
        self.assertEqual(['fscore_binary_0', 'precision_binary_0', 'recall_binary_0'], fbeta.__name__)

    def test_predefined_names(self):
        epoch_metrics = [
            'f1',
            'precision',
            'recall',
            'binaryf1',
            'binf1',
            'binaryprecision',
            'binprecision',
            'binaryrecall',
            'binrecall',
        ]
        fmetric = ['fscore', 'precision', 'recall', 'fscore', 'fscore', 'precision', 'precision', 'recall', 'recall']
        average = ['macro', 'macro', 'macro', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
        names = [
            'fscore_macro',
            'precision_macro',
            'recall_macro',
            'bin_fscore1',
            'bin_fscore2',
            'bin_precision1',
            'bin_precision2',
            'bin_recall1',
            'bin_recall2',
        ]
        model = Model(nn.Linear(10, 2), 'sgd', 'cross_entropy', epoch_metrics=epoch_metrics)
        actual_fmetric = [epoch_metric._metric for epoch_metric in model.epoch_metrics]
        actual_average = [epoch_metric._average for epoch_metric in model.epoch_metrics]
        self.assertEqual(fmetric, actual_fmetric)
        self.assertEqual(average, actual_average)
        self.assertEqual(names, model.epoch_metrics_names)


class FBetaBinaryTest(TestCase):
    def setUp(self):
        # [0, 1, 1, 1, 0, 1]
        self.predictions = torch.Tensor([[0.35, 0.25], [0.1, 0.6], [0.1, 0.6], [0.1, 0.5], [0.2, 0.1], [0.1, 0.6]])
        self.targets = torch.Tensor([0, 0, 1, 0, 1, 0])

        # detailed target state
        self.pred_sum = [2, 4]
        self.true_sum = [4, 2]
        self.true_positive_sum = [1, 1]
        self.true_negative_sum = [1, 1]
        self.total_sum = [6, 6]

        desired_precision = 0.25
        desired_recall = 0.5
        desired_fscore = (
            (2 * desired_precision * desired_recall) / (desired_precision + desired_recall)
            if desired_precision + desired_recall != 0.0
            else 0.0
        )
        self.output = [desired_fscore, desired_precision, desired_recall]

    def test_fbeta_binary(self):
        fbeta = FBeta(average='binary')
        fbeta.update(self.predictions, self.targets)

        # check state
        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), self.pred_sum)
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), self.true_sum)
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        numpy.testing.assert_almost_equal(fbeta._total_sum.tolist(), self.total_sum)
        numpy.testing.assert_almost_equal(fbeta.compute(), self.output)

    def test_fbeta_binary_one_dim_pred(self):
        fbeta = FBeta(average='binary')
        fbeta.update(self.predictions[:, 1:] - self.predictions[:, 0:1], self.targets)

        # check state
        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), self.pred_sum)
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), self.true_sum)
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        numpy.testing.assert_almost_equal(fbeta._total_sum.tolist(), self.total_sum)
        numpy.testing.assert_almost_equal(fbeta.compute(), self.output)

    def test_fbeta_binary_zero_dim_pred(self):
        fbeta = FBeta(average='binary')
        fbeta.update(self.predictions[:, 1] - self.predictions[:, 0], self.targets)

        # check state
        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), self.pred_sum)
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), self.true_sum)
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        numpy.testing.assert_almost_equal(fbeta._total_sum.tolist(), self.total_sum)
        numpy.testing.assert_almost_equal(fbeta.compute(), self.output)

    def test_fbeta_binary_with_return_batch_value(self):
        fbeta = FBeta(average='binary')
        batch_value = fbeta(self.predictions, self.targets)
        epoch_value = fbeta.compute()

        self.assertEqual(batch_value, epoch_value)
