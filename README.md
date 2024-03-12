# AMaze

A lightweight maze navigation task generator for sighted AI agents.

## Mazes

AMaze is primarily a maze generator: its goal is to provide an easy way to 
generate arbitrarily complex (or simple) mazes for agents to navigate in.

Every maze can be described by a unique, human-readable string:

[//]: # (![maze sample_dark]&#40;https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/dark.png#gh-dark-mode-only&#41;)

[//]: # (![maze sample_light]&#40;https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/light.png#gh-light-mode-only&#41;)

<picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/dark.png">
    <img alt="Maze sample"
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

![inputs_discrete_dark](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-0.png#gh-dark-mode-only)
![inputs_discrete_light](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-0.png#gh-light-mode-only)

In the discrete input case, information is provided in an easily intelligible format of fixed size.
According to the cell highlighted in the previous example, an agent following the optimal trajectory 
would receive the following inputs:

![inputs_discrete_example_dark](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-1.png#gh-dark-mode-only)
![inputs_discrete_example_light](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-1.png#gh-light-mode-only)

### Continuous inputs

![inputs_continuous_dark](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-2.png#gh-dark-mode-only)
![inputs_continuous_light](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-2.png#gh-light-mode-only)

For continuous inputs, a raw grayscale image is directly provided to the agent.
It contains wall information on the outer edge, as well as the same temporal information as with the
discrete case (centered pixel on the corresponding border).
The sign, if any, is thus provided a potentially complex image that the agent must parse and understand:

![inputs_continuous_example_dark](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/dark-3.png#gh-dark-mode-only)
![inputs_continuous_example_light](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/agents/light-3.png#gh-light-mode-only)

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

## Development state (todolist)

- No clear distinction between library code (maze generator, viewer, stats extractor, ...)
and user code (experiments, training protocols, results generation, ...).
- Dependency on stable baselines 3, should (will) be optional.

Install with
```
git clone https://github.com/kgd-al/amaze.git
cd amaze
source ../<virtual_environment>/bin/activate
pip install -e .
```

Patch requests are welcome.

## Documentation

Patchy at best. Should be improved upon in the coming months.

## Examples

Same. No nice examples are provided (yet!).
