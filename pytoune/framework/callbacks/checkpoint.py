"""
The source code of this file was copied from the Keras project, and has been
modified.

COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2017, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings

import torch
from .callbacks import Callback

class ModelCheckpoint(Callback):
    def __init__(self, filename, monitor='val_loss', verbose=False, save_best_only=False, mode='min', period=1):
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only

        if self.save_best_only:
            if mode not in ['min', 'max']:
                raise ValueError("Invalid mode '%s'" % mode)
            if mode == 'min':
                self.monitor_op = lambda x,y: x < y
                self.current_best = float('Inf')
            elif mode == 'max':
                self.monitor_op = lambda x,y: x > y
                self.current_best = -float('Inf')
            self.current_best_weights = None

        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0 and not self.save_best_only:
            filename = self.filename.format_map(logs[-1])
            if self.verbose:
                print('Epoch %d: saving model to %s' % (epoch, filename))
            self.model.save_weights(filename)
        if self.save_best_only:
            if self.monitor_op(logs[-1][self.monitor], self.current_best):
                filename = self.filename.format_map(logs[-1])
                if self.verbose:
                    print('Epoch %d: %s improved from %0.5f to %0.5f, saving model to %s'
                          % (epoch, self.monitor, self.current_best, logs[-1][self.monitor], filename))
                self.current_best = logs[-1][self.monitor]
                self.model.save_weights(filename)
