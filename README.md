# AMaze

A lightweight maze navigation task generator for sighted AI agents.

## Mazes

AMaze is primarily a maze generator: its goal is to provide an easy way to 
generate arbitrarily complex (or simple) mazes for agents to navigate in.

Every maze can be described by a unique, human-readable string:

![maze sample](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/dark.png#gh-dark-mode-only)
![maze sample](https://raw.githubusercontent.com/kgd-al/amaze/master/docs/latex/maze/light.png#gh-light-mode-only)

## Agents

## Current state

No clear distinction between library code (maze generator, viewer, stats extractor, ...)
and user code (experiments, training protocols, results generation, ...).
Dependency on stable baselines 3, should (will) be optional.

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
