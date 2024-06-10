---
title: 'AMaze: a benchmark generator for sighted maze-navigating agents'
tags:
  - Python
  - Reinforcement Learning
  - NeurEvolution
  - Benchmark
  - Vision
authors:
  - name: Kevin Godin-Dubois
    orcid: 0009-0002-6033-3555
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Karine Miras
    affiliation: 1
  - name: Anna V. Kononova
    affiliation: 2
affiliations:
 - name: Vrije Universiteit Amsterdam, The Netherlands
   index: 1
 - name: Leiden University, The Netherlands
   index: 2
date: 21 June 2024
bibliography: paper.bib
---

# Summary

The need to provide fair comparisons between agents, especially in the field of Reinforcement Learning, has led to a plethora of benchmarks.
However these benchmarks are, for the most part, devoted to tailor made problems with very little degrees of freedom for the experimenter.
AMaze is, instead, a benchmark *generator* capable of producing human-intelligible environment of arbitrarily high complexity.
By using, potentially custom-made, visual cues in a maze-navigation task, the library empowers researchers across a large breadth of fields.

# Statement of need

AMaze is a pure-python package with an emphasis towards easy and intuitive generation, evaluation and analysis of mazes.
Its primary goal is to provide a way to quickly generate mazes of targeted difficulty e.g. to test a Reinforcement Learning algorithm.
Without paraphrasing the documentation[^1] too much, users of AMaze have two main components to take into consideration: mazes and agents.

## Mazes
Every maze can be described by human-readable string as illustrated in \autoref{fig:maze}, where every component is optional with sensible default values (excepted the *seed* which is time-dependent).
The *seed* is used in the random number generator responsible for: a) the depth-first search that creates the part and b) the stochastic placement of the *lures* and *traps*.
As will be detailed below, agents only see a single cell at a time making intersections impossible to handle without additional information.
*Clues* provide such an information by helpfully pointing towards the correct direction.
However, users may additionally specify the presence of *traps*, at a given frequency, to replace a clue at an intersection.
Traps always point towards the wrong direction (randomly so in case of a three-way intersection) thereby forcing agents to discriminate between the two.
Furthermore, there is a lighter class of negative sign, namely *lures*, which occur outside of intersection and unhelpfully point towards an obviously bad direction (e.g. a wall).

Mazes can broadly be grouped by class depending on the features they exhibit.
The most *trivial* cases correspond to mazes with a single path (enforced by "filling-in" intersections).
When intersections are labeled with appropriate clues mazes are considered as *simple*.
Additionally exhibiting either lures or traps form the corresponding classes while the more general case with all types of signs is labeled as *complex*.
To accurately compare between different types of mazes across multiple categories, the library provides two dedicated metrics.
The surprisingness $S_M$ and deceptiveness $D_M$ for a given maze $M$ are as follows:

![A sample maze from the library. Every maze can be converted to and from a human-readable string where each underscore-separated component describes one of its facets. The *seed* seeds the random number generator used for the paths and stochastic placement of *lures* and *traps*. These have a specific probability, shape and/or value and may be specified multiple times to increase the complexity.\label{fig:maze}](../docs/latex/maze/light-wide.png)

$$S_M = - \sum\limits_{i \in I_M} p(i) * log_2(p(i))$$
$$D_M = \sum\limits_{c \in \text{cells}(M)}
           \sum\limits_{\substack{s \in \text{traps}(M)\\s[0:3] = c}}
            - p(s|c) log_2(p(s|c))$$

which, informally, account for the likelihood of encountering different states (walls, signs) and different *variations* of a given cell (same walls, different signs).
Through this, experimenters can make an informed decision about the level of complexity of the mazes they use.
As illustrated in \autoref{fig:complexity}, the space of all possible mazes[^2] is both diverse and arbitrarily complex.

![Distribution of Surprisingness $S_M$ versus Deceptiveness $D_M$ across 500'000 unique mazes from all five different classes. Outlier mazes are depicted in the borders to illustrate the underlying Surprisingness (right column) or lack thereof (left column).\label{fig:complexity}](../docs/latex/complexity/light.png)

[^1]: [https://amaze.readthedocs.io/en/latest/](https://amaze.readthedocs.io/en/latest/)
[^2]: Sampled from 500'000 unique mazes across all five classes

## Agents

Agents in AMaze are loosely embodied robots that wander around mazes perceiving only local information (the cell they are in) and a single bit of memory (the direction they come from, if any).
To accommodate various use cases, these agents come in three different forms: fully discrete, fully continuous and hybrid.
In the former case, an agent has access to something akin to a pre-processed input, as in \autoref{fig:d}, where the first four fields describes the wall configuration and the remainder provide information about signs, if any.

![Discrete inputs for the examples shown above.\label{fig:d}](../docs/latex/agents/light-1.png){ width=33% }

The direction of the previous cell is depicted in red *for the benefit of the human observer* as agents only perceive grayscale values.
In the fully discrete case, these observations are used to deduce the correct action out of the four cardinal directions.


![Continuous inputs (images) for the examples shown above.\label{fig:c}](../docs/latex/agents/light-3.png){ width=33% }

<!-- ![Discrete.\label{demo:dd}](../docs/demo/dd.gif){ width=33% } -->
<!-- ![Hybrid.\label{demo:cd}](../docs/demo/cd.gif){ width=33% } -->
<!-- ![Continuous.\label{demo:cc}](../docs/demo/cc.gif){ width=33% } -->


Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
