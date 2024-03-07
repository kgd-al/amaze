Using graphics
==============

.. |FILE| replace:: examples/visualization.py

The file |FILE| showcases the use of graphical visualization elements:
    - creating an image from a maze string
    - plotting an agent's trajectory in a maze
    - accessing the main entry point programmatically

.. kgd-literal-include:: 1

As with the simulation-side, all major components are available in a single
import statement.

.. kgd-literal-include:: 3-10
    :emphasize-lines: 8

Next we define some helpful variables including the maze string.
Notice that, in this case, we do not provide a seed to the maze thereby
implying that we want a random one.
Thus, in the highlighted line, we "resolve" the maze string so that the seed
is known.

.. kgd-literal-include:: 13-19

The graphical elements are introduced, first, by creating a QtApplication
object which is essential for PyQt (the underlying widgets library) to work
properly.
Then, rendering a maze to a given file is trivially done by
:meth:`~amaze.visu.widgets.maze.MazeWidget.draw_to`.
The various rendering options which can be provided to tweak the
appearance are detailed in
:meth:`~amaze.visu.widgets.maze.MazeWidget.default_config`.

.. kgd-literal-include:: 21-29
    :emphasize-lines: 7

Then, as in the previous example, we load an agent and have it roaming the maze
until completion.
Note, however, the `save_trajectory` flag provided to the simulation so that
we can later plot it.

.. kgd-literal-include:: 31-36

This is actually done similarly to render an empty maze, although we now
provide the whole simulation and use
:meth:`~amaze.visu.widgets.maze.MazeWidget.plot_trajectory`.

.. kgd-literal-include:: 42

Finally, we also could call the main script from python and provide it with
arguments.
This way one can easily instrumentalize the library, e.g. when computing
generalization performance.
The list of arguments is detailed in :class:`amaze.bin.main.Options`.

As before, the full listing of the example is provided below.

.. kgd-literal-include::

