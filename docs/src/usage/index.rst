Usage
=====

Installation
------------

End-user installation
*********************

The package is written in pure python and is installable via pip.
Optional dependencies (see further down) can be added to provide more functionality:
built-in compatibility with common RL libraries, visualization tools, ...

.. code-block:: console

   (.venv)$ pip install amaze


Tutorials
---------

The reader is first encouraged to play around with the main executable (:mod:`amaze.bin.amaze`).
Without arguments, it will provide an interface where one can manipulate a maze and control an
agent via the keyboard.

.. code-block:: console

    (.venv)$ ???

.. toctree::
    basics
    visualization
    training

FAQ
---

.. warning::

    There is a known deleterious interaction between PyQt5 and opencv with both specifying different plugins.
    The related error is

    .. error::

        QObject::moveToThread: Current thread (...) is not the object's thread (...).
        Cannot move to target thread (..)

        qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in ".../site-packages/cv2/qt/plugins" even though
        it was found.
        This application failed to start because no Qt platform plugin could be initialized.
        Reinstalling the application may fix this problem.

        Available platform plugins are: xcb, ...

    See :class:`amaze.extensions.sb3.utils.CV2QTGuard` as a temporary band aid
