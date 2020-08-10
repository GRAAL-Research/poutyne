.. role:: hidden
    :class: hidden-section

Interface of the ``policy`` module
**********************************

.. note:: See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_interface.ipynb>`_

About the ``policy`` Module Interface
=====================================

The ``policy`` modules give you fine-grained control over the training process.
This example demonstrates how the ``policy`` module works and how you can create your own policies.

Parameter Spaces and Phases
---------------------------

Parameter spaces like ``linspace`` and ``cosinespace`` are the basic building blocks.

.. code-block:: python

    from poutyne.framework import linspace, cosinespace


You can define the space and iterate over them:

.. code-block:: python

    space = linspace(1, 0, 3)
    for i in space:
        print(i)

.. code-block:: python

    space = cosinespace(1, 0, 5)
    for i in space:
        print(i)


You can use the space and create a phase with them:

.. code-block:: python

    from poutyne.framework import Phase

    phase = Phase(lr=linspace(0, 1, 3))

    # and iterate
    for d in phase:
        print(d)


You can also visualize your phase:

.. code-block:: python

    import matplotlib.pyplot as plt
    phase.plot("lr");


Phases can have multiple parameters:

.. code-block:: python

    phase = Phase(
        lr=linspace(0, 1, 10),
        momentum=cosinespace(.99, .9, 10),
    )

    phase.plot("lr");
    phase.plot("momentum")


Visualize Different Phases
--------------------------

.. code-block:: python

    steps = 100

    fig, ax = plt.subplots()
    # Constant value
    Phase(lr=linspace(.7, .7, steps)).plot(ax=ax)
    # Linear
    Phase(lr=linspace(0, 1, steps)).plot(ax=ax)
    # Cosine
    Phase(lr=cosinespace(1, 0, steps)).plot(ax=ax);


Visualize Multiple Parameters in One Phase
------------------------------------------

.. code-block:: python

    steps = 100
    phase = Phase(lr=linspace(1, 0.5, steps), momentum=cosinespace(.8, 1, steps))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    phase.plot("lr", ax=axes[0])
    phase.plot("momentum", ax=axes[1]);


Build Complex Policies From Basic Phases
========================================

You can build complex optimizer policies by chaining phases together:

.. code-block:: python

    from poutyne.framework import OptimizerPolicy

    policy = OptimizerPolicy([
        Phase(lr=linspace(0, 1, 100)),
        Phase(lr=cosinespace(1, 0, 200)),
        Phase(lr=linspace(0, .5, 100)),
        Phase(lr=linspace(.5, .1, 300)),
    ])

    policy.plot();


Use Already Defined Complex Policies
------------------------------------

It's easy to build your own policies, but Poutyne contains some pre-defined phases.

.. code-block:: python

    from poutyne.framework import sgdr_phases

    # build them manually
    policy = OptimizerPolicy([
        Phase(lr=cosinespace(1, 0, 200)),
        Phase(lr=cosinespace(1, 0, 400)),
        Phase(lr=cosinespace(1, 0, 800)),
    ])
    policy.plot()

    # or use the pre-defined one
    policy = OptimizerPolicy(sgdr_phases(base_cycle_length=200, cycles=3, cycle_mult=2))
    policy.plot();


Pre-defined ones are just a list phases:

.. code-block:: python

    sgdr_phases(base_cycle_length=200, cycles=3, cycle_mult=2)


Here is the one-cycle policy:

.. code-block:: python

    from poutyne.framework import one_cycle_phases

    tp = OptimizerPolicy(one_cycle_phases(steps=500))
    tp.plot("lr")
    tp.plot("momentum");


