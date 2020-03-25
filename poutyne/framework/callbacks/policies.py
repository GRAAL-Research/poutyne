"""
The ``policies`` module is an alternative way to configure your training process. It gives you fine
grained control over the process.

The training is divided into phases with the :class:`~poutyne.framework.callbacks.policies.Phase` class.
A :class:`~poutyne.framework.callbacks.policies.Phase` contains parameter spaces (e.g. learning rate,
or momentum, or both) for the optimizer. You chain :class:`~poutyne.framework.callbacks.policies.Phase`
instances by passing them to the :class:`~poutyne.framework.callbacks.policies.OptimizerPolicy`.
:class:`~poutyne.framework.callbacks.policies.OptimizerPolicy` is a :class:`~poutyne.framework.callbacks.Callback`
that uses the phases, steps through them, and sets the parameters of the optimizer.
"""
###############################################################################
import contextlib
from collections import OrderedDict
from itertools import islice, chain
from math import cos, pi
from typing import Dict, List, Tuple, Optional

from .callbacks import Callback


###############################################################################
# Lazy parameter spaces
# A space is just an iterable
class linspace:
    """
    A lazy linear parameter space that goes from ``start`` to ``end`` in ``steps`` steps.

    Args:
        start (int): the start point.
        end (int): the end point.
        steps (int): the number of steps between start and end.

    Example:
        >>> list(linspace(0, 1, 3))
        [0.0, 0.5, 1.0]
    """

    def __init__(self, start: int, end: int, steps: int):
        self.start = start
        self.end = end
        self.steps = steps

    def _progress(self, i):
        return i / (self.steps - 1)

    def __iter__(self):
        return (self.start + self._progress(i) * (self.end - self.start) for i in range(self.steps))


class cosinespace:
    """
    A lazy cosine parameter space that goes from ``start`` to ``end`` in ``steps`` steps.

    Args:
        start (int): the start point.
        end (int): the end point.
        steps (int): the number of steps between start and end.

    Example:
        >>> list(cosinespace(0, 1, 3))
        [0.0, 0.5, 1.0]
    """

    def __init__(self, start: int, end: int, steps: int):
        self.start = start
        self.end = end
        self.steps = steps

    def _progress(self, i):
        return i / (self.steps - 1)

    def __iter__(self):
        return (self.end + (self.start - self.end) * (1 + cos(self._progress(i) * pi)) / 2 for i in range(self.steps))


###############################################################################
class Phase:
    """
    Defines how to configure an optimizer.

    For each train step it returns a dictionary that contains the configuration for the optimizer.

    Args:
        lr (List[float], optional): a configuration space for the learning rate.
        momentum (List[float], optional): a configuration space for the momentum.
    """

    def __init__(self, *, lr: Optional[float] = None, momentum: Optional[float] = None):
        if lr is None and momentum is None:
            raise ValueError("You must specify lr and/or momentum.")

        self.configuration = OrderedDict()
        if lr is not None:
            self.configuration["lr"] = lr
        if momentum is not None:
            self.configuration["momentum"] = momentum

    def __iter__(self):
        names = list(self.configuration.keys())
        values = self.configuration.values()
        for values in zip(*self.configuration.values()):
            yield dict(zip(names, values))

    def __repr__(self):
        return "\n".join([
            "Phase:",
            *["    {}: {}".format(name, val) for name, val in self.configuration.items()],
        ])

    def plot(self, param_name: str = "lr", ax=None):
        """
        Plot the phase for the given `param_name`.

        Args:
            param_name (str, optional): the name of the parameter to plot.
            ax (~matplotlib.pyplot.axis, optional): a matplotlib axis to plot on, if given.

        Returns:
            The matplotlib axis.
        """
        # pylint: disable=import-error
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()

        ax.plot(list(self.configuration[param_name]))
        ax.set_xlabel("steps")
        ax.set_ylabel(param_name)
        return ax


###############################################################################
# complex policies build from simple phases
# pylint
def one_cycle_phases(steps: int,
                     lr: Tuple[float, float] = (0.1, 1),
                     momentum: Tuple[float, float] = (0.95, 0.85),
                     finetune_lr: float = .01,
                     finetune_fraction: float = 0.1) -> List[Phase]:
    """
    The "one-cycle" policy as described in the paper `Super-Convergence: Very Fast Training of
    Neural Networks Using Large Learning Rates <https://arxiv.org/abs/1708.07120>`_.

    You might want to read the paper and adjust the parameters.

    Args:
        steps (int): the total number of steps to take.
        lr (Tuple[float, float]): tuple for the triangular learning rate (start, middle).
        momentum (Tuple[float, float]): tuple for the triangular momentum (start, middle).
        finetune_lr (float): target learning rate for the final fine tuning. Should be smaller than
            `min(lr)`.
        finetune_fraction (float): fraction of steps used for the fine tuning.
            Must be between 0 and 1.

    Returns:
        A list of configured :class:`~poutyne.framework.callbacks.policies.Phase` instances.

    References:
        `Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
        <https://arxiv.org/abs/1708.07120>`_
    """
    steps_annealing = int(steps * finetune_fraction)
    steps_up = (steps - steps_annealing) // 2
    steps_down = steps - steps_annealing - steps_up
    return [
        Phase(
            lr=linspace(lr[0], lr[1], steps_up),
            momentum=linspace(momentum[0], momentum[1], steps_up),
        ),
        Phase(
            lr=linspace(lr[1], lr[0], steps_down),
            momentum=linspace(momentum[1], momentum[0], steps_down),
        ),
        Phase(
            lr=linspace(lr[0], finetune_lr, steps_annealing),
            momentum=linspace(momentum[0], momentum[0], steps_annealing),
        ),
    ]


def sgdr_phases(
        base_cycle_length: int,
        cycles: int,
        lr: Tuple[float, float] = (1., 0.1),
        cycle_mult: int = 2,
) -> List[Phase]:
    """
    The "SGDR" policy as described in the paper `SGDR: Stochastic Gradient Descent with Warm Restarts
    <https://arxiv.org/abs/1608.03983>`_.

    Note the total number of steps is calculated like this: `total_steps = sum(base_cycle_length *
    (cycle_mult ** i) for i in range(cycles))`

    You might want to read the paper and adjust the parameters.

    Args:
        base_cycle_length (int): number of steps for the first cycle.
        cycles (int): the number of repetitions.
        lr (Typle[float, float]): tuple for the learning rate for one cycle: (start, end).
        cycle_mult (float): multiply the last cycle length with this every cycle. The length of a cycle
            grows exponentially.

    Returns:
        A list of configured :class:`~poutyne.framework.callbacks.policies.Phase` instances.

    References:
        `SGDR: Stochastic Gradient Descent with Warm Restarts
        <https://arxiv.org/abs/1608.03983>`_
    """
    steps = [base_cycle_length * (cycle_mult**i) for i in range(cycles)]
    return [Phase(lr=cosinespace(lr[0], lr[1], step)) for step in steps]


###############################################################################
class OptimizerPolicy(Callback):
    """
    Combine different :class:`~poutyne.framework.callbacks.policies.Phase` instances
    in an :class:`~poutyne.framework.callbacks.policies.OptimizerPolicy` and execute the policies in a
    row.

    Args:
        phases (List[~poutyne.framework.callbacks.policies.Phase]):
            A list of :class:`~poutyne.framework.callbacks.policies.Phase` instances.
        initial_step (int): The step to start the policy in. Used for restarting.
    """

    def __init__(self, phases: List, *, initial_step: int = 0):
        super().__init__()
        self.phases = phases
        self.current_step = initial_step
        self.phases_iter = iter(self)

    def on_train_batch_begin(self, batch_number: int, logs: Dict):
        # Don't do anything when we run out of phases.
        with contextlib.suppress(StopIteration):
            spec = next(self.phases_iter)
            self._update_optimizer(spec)

    def __iter__(self):
        space_iter = islice(chain.from_iterable(self.phases), self.current_step, None)
        for param_dict in space_iter:
            self.current_step += 1
            yield param_dict

    def all_steps(self) -> List[Dict]:
        """
        Return the list of dictionaries of configurations for all steps.

        This does not advance the current_step count.

        Returns:
            A list of dictionaries of all the parameters for each step.
        """
        return chain.from_iterable(self.phases)

    def __repr__(self):
        return "OptimizerPolicy:\n    phases: {}\n    current_step: {}".format(self.current_step, len(self.phases))

    def _update_optimizer(self, param_dict: Dict):
        for param_name, param_value in param_dict.items():
            for group in self.model.optimizer.param_groups:
                group[param_name] = param_value

    def plot(self, param_name: str = "lr", ax=None):
        """
        Visualize all  :class:`~poutyne.framework.callbacks.policies.Phase`s of
        this  :class:`~poutyne.framework.callbacks.policies.OptimizerPolicy`.

        Args:
            param_name (str, optional): the name of the parameter to plot.
            ax (~matplotlib.pyplot.axis): a matplotlib axis to plot on, if given.

        Returns:
            The matplotlib axis.
        """
        # pylint: disable=import-error
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()

        values = [step[param_name] for step in self.all_steps()]
        ax.plot(values)
        ax.set_ylabel(param_name)
        ax.set_xlabel("steps")
        return ax
