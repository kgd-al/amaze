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

.. kgd-literal-include:: 10-12

Here, we use the verbose version of the
:class:`~amaze.simu.robot.Robot.BuildData` initializer to also specify what
kind of controller it will use and to provide the necessary parameters.
We rely on the simulation to give the list of possible discrete actions and
set the exploration rate and seed for the controller's random number generator.
The inputs and outputs are specified via the corresponding enumerations instead of single
characters for increased readability.

Training loop
-------------

The training process itself, detailed below, mostly boils down to three things:
    - pick training (and evaluation) maze(s)
    - create a controller
    - simulate a lot of episodes and apply the appropriate training operator

.. kgd-literal-include:: 4-6
    :pyobject: train

This time around, we use the explicit initializer for the
:class:`~amaze.simu.maze.Maze.BuildData`.

.. kgd-literal-include:: 8-10
    :pyobject: train

We then tweak it slightly to get different maze for the agents to be evaluated
in so that we can ensure some small measure of generalized performance.

.. kgd-literal-include:: 12-13
    :pyobject: train

The robot data is used to instantiate one of the builtin controller to which we provide
specific arguments.
Using that same robot data we create a simulation with any one maze.

.. kgd-literal-include:: 17
    :pyobject: train
.. kgd-literal-include:: 28
    :pyobject: train

Then for a certain number of episodes:

.. kgd-literal-include:: 29-30
    :pyobject: train

we let the agent experience a maze and learn from it ...

.. kgd-literal-include:: 36-38
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
    :emphasize-lines: 20

Finally, we illustrate two methods to evaluate the generalization performance of an AMaze
agent.
As we no longer need to explore with this policy, we start by setting epsilon to 0,
ensuring the agent will always take the greedy action.

The first method then consists in generating a large number of random mazes and, for each,
creating a simulation and letting it run until completion.
Thanks to the
:meth:`~amaze.simu.simulation.Simulation.normalized_reward`, we can know if
the agent has followed the optimal trajectory by verifying that it is equal to 1.
By performing this on a large enough sample, we can get a measure of how well the agent
adapts to unseen mazes.

The second method is more straightforward (and computationally cheaper): when inputs are
discrete (either pre-processed with :attr:`~amaze.simu.types.InputType.DISCRETE`/
:attr:`~amaze.simu.types.OutputType.DISCRETE` or aligned images with
:attr:`~amaze.simu.types.InputType.CONTINUOUS`/:attr:`~amaze.simu.types.OutputType.DISCRETE`)
it is possible to actually enumerate all possible combinations.
Such an approach has advantages compared to the more straightforward maze-navigation as a
single error has no potential for catastrophic failure.
At the same time, by being more abstract, it only evaluates the subset of the agents
capabilities responsible for immediate action.
The returned values describe, with various levels of detail, the agents performance.

The main
----------

.. kgd-literal-include::
    :pyobject: main

To tie it all up, the main calls both the training and generalization
functions while also showcasing how to save a fully trained controller.
The :meth:`~amaze.simu.controllers.control.save` function allows for
additional information to be stored alongside the policy's archive for later
retrieval.
