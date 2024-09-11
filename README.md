# AMaze

[![JOSS](https://joss.theoj.org/papers/bc6892f6873808ced10bc8c4c222e635/status.svg)](https://joss.theoj.org/papers/bc6892f6873808ced10bc8c4c222e635)

[![PyPI version](https://badge.fury.io/py/amaze-benchmarker.svg)](https://badge.fury.io/py/amaze-benchmarker)
[![Documentation Status](https://readthedocs.org/projects/amaze/badge/?version=latest)](https://amaze.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/amaze-benchmarker)](https://pepy.tech/project/amaze-benchmarker)

![main:status](https://github.com/kgd-al/amaze/actions/workflows/test_and_deploy.yml/badge.svg?branch=main)
![main:test](https://github.com/kgd-al/amaze/actions/workflows/extensive_testing.yml/badge.svg?branch=main)

[//]: # ([![cov]&#40;https://kgd-al.github.io/amaze/badges/coverage.svg&#41;]&#40;https://github.com/kgd-al/amaze/actions&#41;)

[//]: # (![dev:test]&#40;https://github.com/kgd-al/amaze/actions/workflows/extensive_testing.yml/badge.svg?branch=dev&#41;)

A lightweight maze navigation task generator for sighted AI agents.

## Mazes

AMaze is primarily a maze generator: its goal is to provide an easy way to 
generate arbitrarily complex (or simple) mazes for agents to navigate in.

Every maze can be described by a unique, human-readable string:

<picture>
    <source
        media="(prefers-color-scheme: dark)"
        srcset="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/dark.png">
    <img
        alt="Maze sample"
        src="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/light.png">
</picture>
  

Clues point agents towards the correct direction, which is required for them to solve intersections.
Traps serve the opposite purpose and are meant to provide more challenge to the maze navigation tasks.
Finally, Lures are low-danger traps that can be detected using local information only (i.e. go into a wall).

> **_NOTE:_** The path to solution as well as the finish flag are only visible to the human

## Agents

Agents in AMaze are loosely embodied with only access to local physical information (the current cell)
and one step-removed temporal information (direction of the previous cell, if any).
The input and output spaces can either be discrete or continuous.

### Discrete inputs

<picture>
    <source
        media="(prefers-color-scheme: dark)"
        srcset="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-0.png">
    <img
        alt="Discrete inputs"
        src="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-0.png">
</picture>

In the discrete input case, information is provided in an easily intelligible format of fixed size.
According to the cell highlighted in the previous example, an agent following the optimal trajectory 
would receive the following inputs:

<picture>
    <source
        media="(prefers-color-scheme: dark)"
        srcset="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-1.png">
    <img
        alt="Discrete inputs examples"
        src="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-1.png">
</picture>

### Continuous inputs

<picture>
    <source
        media="(prefers-color-scheme: dark)"
        srcset="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-2.png">
    <img
        alt="Continuous inputs"
        src="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-2.png">
</picture>

For continuous inputs, a raw grayscale image is directly provided to the agent.
It contains wall information on the outer edge, as well as the same temporal information as with the
discrete case (centered pixel on the corresponding border).
The sign, if any, is thus provided as a potentially complex image that the agent must parse and understand:

<picture>
    <source
        media="(prefers-color-scheme: dark)"
        srcset="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-3.png">
    <img
        alt="Continuous inputs"
        src="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-3.png">
</picture>

While the term continuous is a bit of stretch for coarse-grain grayscale images, it highlights the
difference with the discrete case where every possible input is easily enumerable.

### Examples

According to the combinations of input and output spaces, the library can work in one of three ways.

#### Fully discrete

![example_dd](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/demo/dd.gif)

#### Hybrid (continuous inputs, discrete outputs)

![example_dd](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/demo/cd.gif)

#### Fully continuous

![example_cc](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/demo/cc.gif)

## Further reading
The documentation is available at (https://amaze.readthedocs.io/) including
installation instruction (pip) and detailed examples.

## Contributing

Contributions to the library increasing its ease of use in both the scientific and
students communities are more than welcome.
Any such contribution can be made via a pull request while bugs (in the code or the documentation)
should be reported in the dedicated tracker.

As a rule of thumb, you should attempt not to add any additional dependencies.
Currently, the library uses:
- `numpy` and `pandas` for data processing and formatting
- `PyQt5` for the interface

Continuous integration and deployment is handled by github directly through the `test_and_deploy.yml` workflow file.
While not directly runnable, the commands used therein are encapsulated in the generalist script `commands.sh`.
Call it with `-h` to get a list of available instructions.

Before performing a pull request, make sure to run the deployment tests first either through `commands.sh before-deploy` or by calling the dedicated script (`deploy_tests.sh`) directly.
It will install the package under a temporary folder in multiple configurations (user, test, ...) and check for formatting and pep8 compliance.

## Acknowledgement

This research was funded by the Hybrid Intelligence Center, a 10-year programme
funded by the Dutch Ministry of Education, Culture and Science through the
Netherlands Organisation for Scientific Research,
[https://hybrid-intelligence-centre.nl](https://hybrid-intelligence-centre.nl), grant number 024.004.022.
