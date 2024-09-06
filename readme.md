<!--
SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
SPDX-FileCopyrightText: 2024 Tim Hegemann <tim.hegemann@uni-wuerzburg.de>
SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
SPDX-FileCopyrightText: 2024 Alexander Wolff <alexander.wolff@uni-wuerzburg.de>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Graph Harvester

## Overview

Graph Harvester is a program that extracts graphs from drawings of graphs in PDF files. We focus on the problem of extracting graphs from vector data, which is the primary format used in publications nowadays. 

The program can be accessed via [https://go.uniwue.de/graph-harvester](https://go.uniwue.de/graph-harvester).

## Motivation

In graph theory, certain graphs are pivotal in numerous publications, serving as key tools
in proving or refuting theorems. Hence, having a collection of significant graphs previously
used in publications is highly beneficial. This is the idea behind the website [House of Graphs](https://houseofgraphs.org/)
(HoG) set up by Coolsaet et al. [^2]. It explicitly does not aim at making all possible
graphs (of a certain size) available, but only “interesting” ones. On the site, one can search
graphs using many criteria, draw graphs, and upload new graphs. Still, from a graph drawing
perspective, the HoG database is far from being complete. Out of 293 potentially interesting
graphs (with at least seven vertices and maximum vertex degree of at least 3) that we
extracted from figures in papers presented at GD 2023, the HoG database contained only 57.

Unfortunately, extending the HoG collection is time-consuming. Currently, graphs must
be uploaded in one of two formats: either as an adjacency matrix or in graph6 string
representation. HoG users, however, may have their graphs represented differently, e.g., as a
drawing. Converting a drawing to the required formats by hand is prone to errors for smaller
graphs, and practically impossible for larger ones.

## Example

The following image shows an example output of the Graph Harvester website for Figure 8(b) of a paper of Bekos et al. [^1].

![Example output of the Graph Harvester website for Figure 8(b) of a paper of Bekos et al. [1]](/examples/example_bekos_et_al.png)

The website displays a side-by-side view of the graph produced by Graph Harvester to an image of the original figure in the PDF. This allows the user to easily verify the result. The detected graph is also given in graph6 encoding. If the detected graph is found on HoG, a link to the entry is provided. If it is not found, a link to HoG is provided where the user can add or edit the graph.

## Requirements

Currently, parts of the pipeline are closed-source and therefore the only way to use the software is via [https://go.uniwue.de/graph-harvester](https://go.uniwue.de/graph-harvester).





[^1]: Michael A. Bekos, Walter Didimo, Giuseppe Liotta, Saeed Mehrabi, and Fabrizio Montecchiani.
On RAC drawings of 1-planar graphs. Theoret. Comput. Sci., 689:48–57, 2017. doi:10.1016/
j.tcs.2017.05.039.

[^2]: Kris Coolsaet, Sven D’hondt, and Jan Goedgebeur. House of Graphs 2.0: A database of
interesting graphs and more. Discret. Appl. Math., 325:97–107, 2023. doi:10.1016/J.DAM.
2022.10.013.
