Basic usage
===========

.. |FILE| replace:: examples/basics.py

The file |FILE| showcases the major simulation-side components of AMaze:
    - creating a maze, robot and simulation
    - loading from an existing pre-trained agent
    - having the agent explore said maze
    - print various statistics about the maze and the agent's performance

.. kgd-literal-include:: 3

We start by importing modules: everything related to the simulation is conveniently exposed by the module
:mod:`amaze.simu`.

.. note:: :mod:`pprint` is imported to generate pretty outputs and is not mandatory.

.. kgd-literal-include:: 6-8

With every relevant class in local scope we can create the basic components: the
:class:`~amaze.simu.maze.Maze`, :class:`~amaze.simu.robot.Robot` and
:class:`~amaze.simu.simulation.Simulation`.
The first can be quickly generated from a string
(see :meth:`Maze.from_string() <amaze.simu.maze.Maze.from_string>`).
For convenience, the inputs and outputs spaces can also be provided as a string
(:meth:`Robot.BuildData.from_string() <amaze.simu.robot.Robot.BuildData.from_string>`).
The last component is the simulation object, created using both the maze and robot description.

.. kgd-literal-include:: 10

For the curious, line 10 will provide detailed statistics about the maze including its size,
intersections, complexity, ...

.. kgd-literal-include:: 13-14

Next we retrieve one of the demonstration agents provided with the library and print some of the
information stored alongside it.

.. kgd-literal-include:: 16

After which we let the simulation run freely until completion.
This means either :meth:`~amaze.simu.simulation.Simulation.success` (the agent has reached the target) or
:meth:`~amaze.simu.simulation.Simulation.failure` (the agent has exceeded the deadline).

.. kgd-literal-include:: 19-22

To conclude, we extract select pieces of information from the simulation (success, rewards) as well
as the more exhaustive :meth:`~amaze.simu.simulation.Simulation.infos`.
The full listing for the exemple is shown below

.. kgd-literal-include::
