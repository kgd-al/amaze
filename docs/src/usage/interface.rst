Custom interfaces
=================

.. |FILE| replace:: examples/interface.py

The file |FILE| showcases the use of the maze widget to create a custom
interface.

.. kgd-literal-include:: 1-4

As usual we start by importing the required package.
Here, however, we will explicitly qualify every amaze members to better
distinguish from class imported from PyQT (which all start with a `Q`).

.. kgd-literal-include:: 1
    :pyobject: MainWindow

To make this page understandable, we define a class holding everything
together.
The `main`, presented at the bottom will only have to create our custom class
and display it.

.. kgd-literal-include:: 1-
    :pyobject: MainWindow.__init__

First, the constructor delegates the bulk of creating a top-level widget to the
PyQT library.
We, then, create a horizontal layout to lay multiple items next to one another.
In this primitive interface, we place a
:class:`~amaze.visu.widgets.maze.MazeWidget` on the left while a secondary
layout will hold the configuration widgets.

.. kgd-literal-include:: 1-
    :pyobject: MainWindow._create_widgets

We then create specific widgets to customize the maze thanks to the helper
function `_add`.
It's job is to instantiate the widget and add it to the layout.
The :class:`~PyQt5.QtWidgets.QFormLayout` is a special case of vertical layout
that places, next to one another, a widget and its string label.
If the newly created widget needs further configuration we call the provided
function to set up things like range or content.
We then connect the widget's signal to our `reset_maze` function so that
whenever the user inputs new values, the maze is changed accordingly.
Finally, we store everything for future reference.

For this interface we provide 5 configurable options of various types:

* Two :class:`~PyQt5.QtWidgets.QSpinBox` will provide integer input in a given
  range
* One :class:`~PyQt5.QtWidgets.QDoubleSpinBox` provides the same functionality
  but for float values
* A :class:`~PyQt5.QtWidgets.QCheckBox` allows binary input
* A :class:`~PyQt5.QtWidgets.QComboBox` lets the user choose from amongst a set
  of strings

.. kgd-literal-include:: 1-
    :pyobject: MainWindow.reset_maze

The function used to reset the maze is rather trivial as it only fetches values
from the interface to populate a :class:`~amaze.simu.maze.Maze.BuildData`
dataclass.
However it also demonstrates of the specific widgets we used expose their
values programatically: `value` for (Double)SpinBox, `isChecked` for CheckBox and
`currentText` for ComboBox.

.. kgd-literal-include:: 1-
    :pyobject: main

Finally, as stated above, the consists only of creating an application and our
primitive main window, requesting it to be shown and letting PyQT handle the
rest.

As before, the full listing of the example is provided below.

.. kgd-literal-include::

