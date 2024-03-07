Stable baselines 3
==================

.. |FILE| replace:: examples/extensions/sb3.py

Training
--------

In this example, we showcase how the built-in stable baselines 3 (sb3)
extension can be used to smoothly leverage the large associated
collection of algorithm and policies.

.. kgd-literal-include:: 1-22
    :emphasize-lines: 10-13

As usual, we start by importing the necessary packages and we define some global
configuration options.
Note that, in addition to the traditional amaze classes, we also import
extension-specific items (detailed below).

.. kgd-literal-include:: 1-4
    :pyobject: train

The training function is defined much more shortly than in the hand-written
q-learning case thanks to the added functionalities of stable baselines 3 and
added wrappers.
While, creating mazes and robots should be familiar by now, we see a new
extension-specific function
:meth:`~amaze.extensions.sb3.maze_env.make_vec_maze_env`
used to create Vectorized Environments
(:class:`~stable_baselines3.common.vec_env.VecEnv`)

.. kgd-literal-include:: 6-7
    :pyobject: train

We also, sometimes, need access to the underlying environments (regular mazes) as
illustrated below.
There we collect the average optimal reward by calling
:meth:`~amaze.extensions.sb3.maze_env.MazeEnv.optimal_reward` on every maze
used for intermediate performance evaluation thanks to
:meth:`~amaze.extensions.sb3.maze_env.env_method`.

.. kgd-literal-include:: 9-23
    :pyobject: train

Next we create a
:class:`~amaze.extensions.sb3.callbacks.TensorboardCallback`, an illustrative
built-in callback that uses Tensorboard to provide an overview of the training
process.
In addition to logging numerical data such as the average rewards it also
automatically generates trajectory images whenever the
:class:`~stable_baselines3.common.callbacks.EventCallback` is triggered.
The following lines define such an object, in a traditional SB3 fashion, while
adding our own tensorboard callback and also using the optimal reward to stop
as soon as the agent is behaving optimally.

.. kgd-literal-include:: 25-33
    :pyobject: train

Finally, we create the sb3 model, using the dedicated wrapper
:meth:`~amaze.extensions.sb3.sb3_controller`, by providing, first, the type
of underlying model (one of :meth:`~amaze.extensions.sb3.compatible_models`)
and, afterwards, the usual parameters.
Then after setting up the logger and letting the training process run its
course, we perform a final step of the callback to render the final
trajectories.

Using
-----

.. kgd-literal-include:: 1-2
    :pyobject: evaluate

Once the training process is complete, we evaluate the resulting agent's
generalization capability in the same manner as in :doc:`training`.
The only difference is the use of the dedicated loading function
:meth:`~amaze.extensions.sb3.load_sb3_controller` which is a verbose alias to
:meth:`~amaze.simu.controllers.control.load`.
The reminder of this function being the same, we refer the reader to the
previous example, if needed.

.. kgd-literal-include::
    :pyobject: main

Finally, the main should also be familiar from the previous example.
One thing to note, however, is that, due to incompatibilities between the current
opencv and PyQT5 libraries, one should use
:class:`~amaze.extensions.sb3.guard.CV2QTGuard` when combining stable baselines
3 with the native Qt5 components.
