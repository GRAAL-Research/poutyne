# v1.x.x

*

# v1.3.1

* Bug fix for when changing the GPU device twice with optimizer having a state would crash.

# v1.3

* A progress bar is now set on validation a model (similar to training). It is disableable by passing
`progress_options=dict(show_on_valid=False)` in the `fit*` methods.
* A progress bar is now set testing a model (similar to training). It is disableable by passing `verbose=False` in the
`evaluate*` methods.
* A new notification callback [`NotificationCallback`](https://poutyne.org/callbacks.html#poutyne.NotificationCallback)
  allowing to received message at specific time (start/end training/testing an at any given epoch).
* A new logging callback, [`MLflowLogger`](https://poutyne.org/callbacks.html#poutyne.MLFlowLogger), this callback allows
  you to log experimentation configuration and metrics during training, validation and testing.
* Fix bug where [`evaluate_generator`](https://poutyne.org/model.html#poutyne.Model.evaluate_generator) did not support
  generators with StopIteration exception.
* Experiment now has a [`train_data`](https://poutyne.org/experiment.html#poutyne.Experiment.train_data) and a
  [`test_data`](https://poutyne.org/experiment.html#poutyne.Experiment.test_data) method.
* The [Lambda layer](https://poutyne.org/layers.html#poutyne.Lambda) now supports multiple arguments in its forward method.

# v1.2

* A `device` argument is added to [`Model`](https://poutyne.org/model.html#poutyne.Model).
* The argument `optimizer` of [`Model`](https://poutyne.org/model.html#poutyne.Model) can now be a dictionary. This
  allows to pass different argument to the optimizer, e.g. `optimizer=dict(optim='sgd', lr=0.1)`.
* The progress bar now uses 20 characters instead of 25.
* The progress bar is now more fluid since partial blocks are used allowing increments of 1/8th of a block at once.
* The function [`torch_to_numpy`](https://poutyne.org/utils.html#poutyne.torch_to_numpy) now does .detach() before
  .cpu(). This might slightly improves performances in some cases.
* In Experiment, the [`load_checkpoint`](https://poutyne.org/experiment.html#poutyne.Experiment.load_checkpoint) method
  can now load arbitrary checkpoints by passing a filename instead of the usual argument.
* Experiment now has a `train_dataset` and a `test_dataset` method.
* Experiment is not considered a beta feature anymore.

**Breaking changes:**

* In [`evaluate`](https://poutyne.org/model.html#poutyne.Model.evaluate), `dataloader_kwargs` is now a dictionary
  keyword argument instead of arbitrary keyword arguments. Other methods are already this way. This was an oversight of
  the last release.

# v1.1

* There is now a batch metric [`TopKAccuracy`](https://poutyne.org/metrics.html#poutyne.TopKAccuracy) and it is possible
  to use them as strings for `k` in 1 to 10 and 20, 30, …, 100, e.g. `'top5'`.
* Add [`fit_dataset`](https://poutyne.org/model.html#poutyne.Model.fit_dataset)
  ,  [`evaluate_dataset`](https://poutyne.org/model.html#poutyne.Model.evaluate_dataset)
  and  [`predict_dataset`](https://poutyne.org/model.html#poutyne.Model.predict_dataset) methods which allow to pass
  PyTorch Datasets and creates DataLoader internally. Here
  is [an example with MNIST](https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_mnist_classification.py)
  .
* Colors now work correctly in Colab.
* The default colorscheme was changed so that it looks good in Colab, notebooks and command line. The previous one was
  not readable in Colab.
* Checkpointing callbacks now don't use the Python [`tempfile` package](https://docs.python.org/3/library/tempfile.html)
  anymore for the temporary file. The use of this package caused problem when the temp filesystem was not on the same
  partition as the final destination of the checkpoint. The temporary file is now created at the same place as the final
  destination. Thus, in most use cases, this will render the use of the `temporary_filename` argument not necessary. The
  argument is still available for those who need it.
* In Experiment, it is not possible to call the method `test`  when training without logging.

# v1.0.1

Update following bug in new PyTorch version: https://github.com/pytorch/pytorch/issues/47007

# v1.0.0

## Version 1.0.0 of Poutyne is here!

* Output is now very nicely colored and now has a progress bar. Both are disableable with the ``progress_options``
  arguments. The ``colorama`` package needs to be installed to have the colors. See the documentation of
  the [fit](https://poutyne.org/model.html#poutyne.Model.fit) method for details.
* Multi-GPU support: Uses ``torch.nn.parallel.data_parallel`` under the hood.
* Huge update to the documentation with a documentation of metrics and a lot of examples.
* No need to import ``framework`` anymore. Everything now can be imported from ``poutyne``directly,
  i.e. ``from poutyne import whatever_you_want``.
* [``PeriodicSaveCallbacks``](https://poutyne.org/callbacks.html#poutyne.PeriodicSaveCallback) (such
  as [``ModelCheckpoint``](https://poutyne.org/callbacks.html#poutyne.ModelCheckpoint)) now has a
  flag ``keep_only_last_best`` which allow to only keep the last best checkpoint even when the names differ between
  epochs.
* [``FBeta``](https://poutyne.org/metrics.html#poutyne.FBeta) now supports an ``ignore_index`` as
  in ``nn.CrossEntropyLoss``.
* Epoch metrics strings ``'precision'`` and ``'recall'`` now available directly without instantiating ``FBeta``.
* Better ETA estimation in output by weighting more recent batches than older batches.
* Batch metrics [``acc``](https://poutyne.org/metrics.html#poutyne.acc)
  and [``bin_acc``](https://poutyne.org/metrics.html#poutyne.bin_acc) now have class
  counterparts [``Accuracy``](https://poutyne.org/metrics.html#poutyne.Accuracy)
  and [``BinaryAccuracy``](https://poutyne.org/metrics.html#poutyne.BinaryAccuracy) in addition to a ``reduction``
  keyword argument as in PyTorch.
* Various bug fixes.

# v0.8.2

- Add new callback methods `on_test_*` to callbacks. Callback can now be passed to the `evaluate*` methods.
- New epoch metrics for scikit-learn functions (
  See [documentation of SKLearnMetrics](https://poutyne.org/metrics.html#poutyne.framework.metrics.SKLearnMetrics)).
- It is now possible to return multiple metrics for a single batch metric function or epoch metric object. Furthermore,
  their names can be changed. (See note
  in [documentation of Model class](https://poutyne.org/model.html#poutyne.framework.Model))
- Computation of batch size is now added for dictionnary inputs and outputs. (
  See [documentation of the new method `get_batch_size`](https://poutyne.org/model.html#poutyne.framework.Model.get_batch_size))
- Add a lot of type hinting.

**Breaking changes:**

- Ground truths and predictions returned by `evaluate_generator` and `predict_generator` are going to be concatenated
  except when inside custom objects in the next version. A warning is issued in those methods. If the warning is
  disabled as instructed, the new behavior takes place. (See documentation
  of [evaluate_generator](https://poutyne.org/model.html#poutyne.framework.Model.evaluate_generator)
  and [predict_generator](https://poutyne.org/model.html#poutyne.framework.Model.predict_generator))
- Names of methods `on_batch_begin` and `on_batch_end` changed to  `on_train_batch_begin` and `on_train_batch_end`
  respectively. When the old names are used, a warning is issued with backward compatibility added. This backward
  compatibility will be removed in the next version.
- `EpochMetric` classes now have an obligatory reset method.
- Support of Python 3.5 is dropped. (Anyway, PyTorch was already not supporting it either)

# v0.7.2

## Poutyne is now under LGPLv3 instead of GPLv3.

Essentially, what this means is that you can now include Poutyne into any proprietary software as long as you are
willing to provide the source code and the modifications of Poutyne with your software. The LICENSE file contains more
details.

This is not legal advice. You should consult your lawyer about the implication of the license for your own case.

# v0.7.1

* Fix a bug introduced in v0.7 when only one of epoch metrics and batch metrics were provided and we would try to
  concatenate a tuple and a list.

# v0.7

* Add automatic naming for class object in `batch_metrics` and `epoch_metrics`.
* Add get_saved_epochs method to Experiment
* `optimizer` parameter can now be set to None in `Model`in the case where there is no need for it.
* Fixes warning from new PyTorch version.
* Various improvement of the code.

*Breaking changes:*

* Threshold of the binary_accuracy metric is now 0 instead of 0.5 so that it works using the logits instead of the
  probabilities.
* The attribute `model` of the `Model` class is now called `network` instead. A deprecation warning is in place until
  the next version.

# v0.6

* Poutyne now has a new logo!
* Add a beta `Experiment` class that encapsulates logging and checkpointing callbacks so that it is possible to stop and
  resume optimization at any time.
* Add epoch metrics allowing to compute metrics over an epoch that are not decomposable such as F1 scores, precision,
  recall. While only these former epoch metrics are currently available in Poutyne, epoch metrics can allow to compute
  the AUROC metric, PCC metric, etc.
* Support for multiple batches per optimizer step. This allows to have smaller batches that fit in memory instead of a
  big batch that does not fit while retaining the advantage of the big batch.
* Add return_ground_truth argument to evaluate_generator.
* Data loading is now taken into account time for progress estimation.
* Various doc updates and example finetunings.

*Breaking changes:*

* `metrics` argument in Model is now deprecated. This argument will be removed in the next version. Use `batch_metrics`
  instead.
* `pytoune` package is now removed.
* If steps_per_epoch or validation_steps are greater than the generator length in *_generator methods, then the
  generator is cycled through instead of stopping as before.

# v0.5.1

* Update for PyTorch 1.1.
* Transfers metric modules to GPU when appropriate.

# v0.5

* Adding a new `OptimizerPolicy` class allowing to have Phase-based learning rate policies. The two following learning
  policies are also provided:
    * "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates", Leslie N. Smith, Nicholay
      Topin, https://arxiv.org/abs/1708.07120
    * "SGDR: Stochastic Gradient Descent with Warm Restarts", Ilya Loshchilov, Frank
      Hutter,  https://arxiv.org/abs/1608.0398
* Adding of "bin_acc" metric for binary classification in addition to the "accuracy" metric".
* Adding "time" in callbacks' logs.
* Various refactoring and small bug fixes.

# v0.4.1

Breaking changes:

- Update for PyTorch 0.4.1 (PyTorch 0.4 not supported)
- Keyword arguments must now be passed with their keyword names in most PyToune functions.

Non-breaking changes:

- self.optimizer.zero_grad() is called instead of self.model.zero_grad().
- Support strings as input for all PyTorch loss functions, metrics and optimizers.
- Add support for generators that raise the StopIteration exception.
- Refactor of the Model class (no API break changes).
- Now using pylint as code style linter.
- Fix typos in documentation.

# v0.4

- New usage example using MNIST
- New *_on_batch methods to Model
- Every Numpy array is converted into a tensor and vice-versa everywhere it applies i.e. methods return Numpy arrays and
  can take Numpy arrays as input.
- New convenient simple layers (Flatten, Identity and Lambda layers)
- New callbacks to save optimizers and LRSchedulers.
- New Tensorboard callback.
- Various bug fixes and improvements.

# v0.3

**Breaking changes**:

- Update to PyTorch 0.4.0
- When one or zero metric is used, evaluate and evaluate generator do not return numpy arrays anymore.

Other changes:

- Model now offers a to() method to send the PyTorch module and its input to a specified device. (thanks PyTorch 0.4.0)
- There is now a 'accuracy' metric that can be used as string in the metrics list.
- Various bug fixes.

# v0.2.2

Last release before an upgrade with breaking changes due to the update of PyTorch 0.4.0.

* Add an on_backward_end callback function
* Add a ClipNorm callback
* Fix various bugs.

# v0.2.1

* Fix warning bugs and bad logic in checkpoints.
* Fix bug where we did not display metric when its value was equal to zero.

# v0.2

* ModelCheckpoint now writes off the checkpoint atomically.
* New initial_epoch parameter to Model.
* Mean of losses and metrics done with batch size weighted by len(y) instead of just the mean of the losses and metrics.
* Update to the documentation.
* Model's predict and evaluate makes more sense now and have now a generator version.
* Few other bug fixes.

# v0.1.1

Doc update

# v0.1

Initial version
