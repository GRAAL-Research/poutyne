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

import os
from io import BytesIO
from tempfile import TemporaryDirectory
from unittest import TestCase
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image

from poutyne import plot_history, plot_metric

mpl.use('Agg')


class PlotHistoryTest(TestCase):
    HISTORY = [
        {'epoch': 1, 'time': 6.2788, 'loss': 0.3683, 'acc': 88.2645, 'val_loss': 0.0984, 'val_acc': 97.0833},
        {'epoch': 2, 'time': 6.2570, 'loss': 0.1365, 'acc': 95.9166, 'val_loss': 0.0680, 'val_acc': 97.9916},
        {'epoch': 3, 'time': 6.3314, 'loss': 0.1057, 'acc': 96.8458, 'val_loss': 0.0531, 'val_acc': 98.4916},
        {'epoch': 4, 'time': 6.2574, 'loss': 0.0874, 'acc': 97.2937, 'val_loss': 0.0521, 'val_acc': 98.4166},
        {'epoch': 5, 'time': 6.3268, 'loss': 0.0775, 'acc': 97.6666, 'val_loss': 0.0464, 'val_acc': 98.6166},
    ]
    METRICS = ['time', 'loss', 'acc']
    NUM_METRIC_PLOTS = len(METRICS)

    def test_basic(self):
        figs, axes = plot_history(PlotHistoryTest.HISTORY)
        self.assertEqual(len(figs), PlotHistoryTest.NUM_METRIC_PLOTS)
        self.assertEqual(len(axes), PlotHistoryTest.NUM_METRIC_PLOTS)
        for fig, ax in zip(figs, axes):
            self.assertIsInstance(fig, Figure)
            self.assertIsInstance(ax, Axes)

    def test_compare_plot_history_plot_metric(self):
        plot_history_figs, _ = plot_history(PlotHistoryTest.HISTORY, close=False, show=False)

        plot_metrics_figs = []
        for metric in PlotHistoryTest.METRICS:
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, ax=ax)
            plot_metrics_figs.append(fig)

        for plot_history_fig, plot_metrics_fig in zip(plot_history_figs, plot_metrics_figs):
            self.assertEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig))

    def test_all_different(self):
        plot_history_figs, _ = plot_history(PlotHistoryTest.HISTORY, close=False, show=False)
        images = list(map(self._to_image, plot_history_figs))
        for i, image_i in enumerate(images):
            for j in range(i + 1, len(images)):
                self.assertNotEqual(image_i, images[j])

    def test_title(self):
        title = 'My Title'
        plot_history_figs, _ = plot_history(PlotHistoryTest.HISTORY, titles=title, close=False, show=False)

        plot_metrics_figs_no_title = []
        for metric in PlotHistoryTest.METRICS:
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, ax=ax)
            plot_metrics_figs_no_title.append(fig)

        plot_metrics_figs_with_title = []
        for metric in PlotHistoryTest.METRICS:
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, title=title, ax=ax)
            plot_metrics_figs_with_title.append(fig)

        for (
            plot_history_fig,
            plot_metrics_fig_no_title,
            plot_metrics_fig_with_title,
        ) in zip(plot_history_figs, plot_metrics_figs_no_title, plot_metrics_figs_with_title):
            self.assertEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig_with_title))
            self.assertNotEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig_no_title))

    def test_different_titles(self):
        titles = ['Time', 'Loss', 'Accuracy']
        plot_history_figs, _ = plot_history(PlotHistoryTest.HISTORY, titles=titles, close=False, show=False)

        plot_metrics_figs_no_title = []
        for metric in PlotHistoryTest.METRICS:
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, ax=ax)
            plot_metrics_figs_no_title.append(fig)

        plot_metrics_figs_with_title = []
        for metric, title in zip(PlotHistoryTest.METRICS, titles):
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, title=title, ax=ax)
            plot_metrics_figs_with_title.append(fig)

        for (
            plot_history_fig,
            plot_metrics_fig_no_title,
            plot_metrics_fig_with_title,
        ) in zip(plot_history_figs, plot_metrics_figs_no_title, plot_metrics_figs_with_title):
            self.assertEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig_with_title))
            self.assertNotEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig_no_title))

    def test_labels(self):
        labels = ['Time', 'Loss', 'Accuracy']
        plot_history_figs, _ = plot_history(PlotHistoryTest.HISTORY, labels=labels, close=False, show=False)

        plot_metrics_figs_no_labels = []
        for metric in PlotHistoryTest.METRICS:
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, ax=ax)
            plot_metrics_figs_no_labels.append(fig)

        plot_metrics_figs_with_labels = []
        for metric, label in zip(PlotHistoryTest.METRICS, labels):
            fig, ax = plt.subplots()
            plot_metric(PlotHistoryTest.HISTORY, metric, label=label, ax=ax)
            plot_metrics_figs_with_labels.append(fig)

        for (
            plot_history_fig,
            plot_metrics_fig_no_labels,
            plot_metrics_fig_with_labels,
        ) in zip(plot_history_figs, plot_metrics_figs_no_labels, plot_metrics_figs_with_labels):
            self.assertEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig_with_labels))
            self.assertNotEqual(self._to_image(plot_history_fig), self._to_image(plot_metrics_fig_no_labels))

    def test_plot_metrics_use_gca(self):
        fig, ax = plt.subplots()
        plot_metric(PlotHistoryTest.HISTORY, 'loss', ax=ax)
        image_with_provided_ax = self._to_image(fig)

        plot_metric(PlotHistoryTest.HISTORY, 'loss')
        image_with_gca = self._to_image(plt.gcf())

        self.assertEqual(image_with_provided_ax, image_with_gca)

    def test_with_provided_axes(self):
        provided_figs, provided_axes = zip(*(plt.subplots() for _ in range(PlotHistoryTest.NUM_METRIC_PLOTS)))
        ret_figs, ret_axes = plot_history(PlotHistoryTest.HISTORY, close=False, show=False, axes=provided_axes)

        new_figs, _ = plot_history(PlotHistoryTest.HISTORY, close=False, show=False)

        self.assertEqual(ret_axes, provided_axes)
        self.assertEqual(len(ret_figs), 0)

        for provided_fig, new_fig in zip(provided_figs, new_figs):
            self.assertEqual(self._to_image(provided_fig), self._to_image(new_fig))

    def test_save(self):
        temp_dir_obj = TemporaryDirectory()
        path = temp_dir_obj.name
        filename = 'test_{metric}'
        final_png_filenames = [
            os.path.join(path, f'{filename.format(metric=metric)}.png') for metric in PlotHistoryTest.METRICS
        ]
        final_pdf_filenames = [
            os.path.join(path, f'{filename.format(metric=metric)}.pdf') for metric in PlotHistoryTest.METRICS
        ]

        figs, _ = plot_history(
            PlotHistoryTest.HISTORY,
            close=False,
            show=False,
            save=True,
            save_directory=path,
            save_filename_template=filename,
            save_extensions=('png', 'pdf'),
        )

        for png_filename, pdf_filename in zip(final_png_filenames, final_pdf_filenames):
            self.assertTrue(os.path.isfile(png_filename))
            self.assertTrue(os.path.isfile(pdf_filename))

        saved_images = [Image.open(filename) for filename in final_png_filenames]
        ret_images = [self._to_image(fig) for fig in figs]
        for save_image, ret_image in zip(saved_images, ret_images):
            self.assertEqual(save_image, ret_image)

    def _to_image(self, fig):
        fd = BytesIO()
        fig.savefig(fd, format='png')
        image = Image.open(fd)
        plt.close(fig)
        return image
