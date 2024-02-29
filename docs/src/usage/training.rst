Full exemple: Training
======================

.. |FILE| replace:: examples/q_learning.py

This example is a full training process for a very basic agent capable of
navigating trivial mazes.
Under the hood, it uses a
:class:`~amaze.simu.controllers.tabular.TabularController` to map discrete
states to discrete actions.
Only the most important pieces of the code will be presented here, with the
reader being redirected to the |FILE| for the unabridged sources.

Configuration
-------------

.. kgd-literal-include::
    :pyobject: robot_build_data

Here, we use the verbose version of the
:class:`~amaze.simu.robot.Robot.BuildData` initializer to also specify what
kind of controller it will use and to provide the necessary parameters.
We rely on the simulation to give the list of possible discrete actions and
set the exploration rate and seed for the controller's random number generator.
The inputs and outputs are specified via the enumeration instead of single
characters for increase readability.

Training loop
-------------

The training process itself, detailed below, mostly boils down to three things:
- pick training (and evaluation) maze(s)
- create a controller
- simulate a lot of episodes and apply the appropriate training operator

.. kgd-literal-include:: 1-14
    :pyobject: train

This time around, we use the explicit initializer for the
:class:`~amaze.simu.maze.Maze.BuildData`.

.. kgd-literal-include:: 16-21
    :pyobject: train

We then tweak it slightly to get different maze for the agents to be evaluated
in so that we can ensure some small measure of generalized performance.

.. kgd-literal-include:: 23-28
    :pyobject: train

We then use the robot data to instantiate one of the builtin controller and
we confirm that the inputs / outputs are compatible with the controller (
:meth:`~amaze.simu.controllers.base.BaseController.inputs_types`,
:meth:`~amaze.simu.controllers.base.BaseController.outputs_types`).
Using that same robot we create a simulation with any one maze.

.. kgd-literal-include:: 32
    :pyobject: train
.. kgd-literal-include:: 42
    :pyobject: train

Then for a certain number of episodes:

.. kgd-literal-include:: 43-45
    :pyobject: train

we let the agent experience a maze and learn from it ...

.. kgd-literal-include:: 49-56
    :pyobject: train

... while also monitoring its performance on unseen mazes.

Learning
--------

.. kgd-literal-include::
    :pyobject: q_train

In the training process, we can no longer use the helpful
:meth:`~amaze.simu.simulation.Simulation.run` function to encapsulate everything as we need
to correlate actions to rewards.
Instead we apply the policy to the current state to get an action.
This action is then used to
:meth:`~amaze.simu.simulation.Simulation.step` the simulation, resulting in a
reward that we can feed back to the policy.
The builtin :class:`~amaze.simu.controllers.tabular.TabularController` has
both sarsa and q-learning natively implemented the latter being used here to
drive the learning process.

Evaluating
----------

.. kgd-literal-include::
    :pyobject: q_eval

In essence, evaluating the performance of an agent on non-training mazes is
very similar to the training process except that we make sure to never use
exploration.
Thus we instead ask the tabular policy to only use
:meth:`~amaze.simu.controllers.tabular.TabularController.greedy_action`.

Generalization
--------------

.. kgd-literal-include::
    :pyobject: evaluate_generalization

Finally, in the context of training generalized agents, we illustrate how to
easily evaluate on a large range of mazes.
As we no longer need to explore with this policy, we start by setting
epsilon to 0, ensuring the agent will always take the greedy action.
Then, as previously, we generate a maze (here randomly), create a simulation
and let it run until completion.
Thanks to the
:meth:`~amaze.simu.simulation.Simulation.normalized_reward`, we can know if
the agent has followed the optimal trajectory by verifying that it is equal to
1.
This makes it easy to ascertain if the agent is indeed performing adequately,
even on unseen mazes.

The main
----------

.. kgd-literal-include::
    :pyobject: main

To tie it all up, the main calls both the training and generalization
functions while also showcasing how to save a fully trained controller.
The :meth:`~amaze.simu.controllers.control.save` function allows for
additional information to be stored alongside the policy's archive for later
retrieval.
