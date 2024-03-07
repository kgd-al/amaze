Stable baselines 3
==================

.. |FILE| replace:: examples/extensions/sb3.py

In this example, we showcase how the built-in stable baselines 3 (sb3)
extension can be used to smoothly leverage the large associated
collection of algorithm and policies.

.. kgd-literal-include:: 1-18
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

We also, sometimes need access to the underlying environments (regular mazes) as
illustrated below.
There we collect the average optimal reward by calling
:meth:`~amaze.extensions.sb3.maze_env.MazeEnv.optimal_reward` on every maze
used for intermediate performance evaluation thanks to
:meth:`~amaze.extensions.sb3.maze_env.env_method`.

.. kgd-literal-include:: 9-23
    :pyobject: train



.. kgd-literal-include::
