<!--
SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
SPDX-FileCopyrightText: 2024 Tim Hegemann <tim.hegemann@uni-wuerzburg.de>
SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
SPDX-FileCopyrightText: 2024 Alexander Wolff <alexander.wolff@uni-wuerzburg.de>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Graph Harvester


## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Example](#example)
- [GraphHarvester API](#graphharvester-api)
  - [Endpoints](#endpoints)
  - [Graph JSON Structure](#graph-json-structure)
  - [Initialization](#initialization)


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

## GraphHarvester API
This API provides endpoints to upload a PDF or process an arXiv link, extracting and returning structured graph data from figures within the document.
## Endpoints
### `POST /upload`

- **Description**: Upload a PDF file, process it to extract graph data, and return the result.
- **Request**: Multipart form-data with a single file parameter:
  ```http
  POST /upload
  Content-Type: multipart/form-data
  ```
  - **file**: The PDF file to process.

- **Response**: JSON array containing graph data extracted from figures within the PDF (see [Figure JSON Structure](#graph-json-structure) below).

---

## Graph JSON Structure

The API returns an array of figure objects, each representing extracted graph data. Each figure object contains several fields that describe the shapes, graphs, and metadata associated with the figure.

### Fields

Each object in the JSON array is a graph, where each subgraph has the following fields:

| Field           | Type                       | Description |
|-----------------|----------------------------|-------------|
| `graph6_string`| `str`              | String representing graph encoded in [graph6 format](https://en.wikipedia.org/wiki/Graph6). |
| `circles`       | `list`            | List representing circles used in graphs. Each inner list corresponds to circles, where each circle has: `c` (Point): Center, `r` (float): Radius, `i` (int): index. |
| `rects`         | `list`            | List representing rectangles, each with properties: `topLeft` (Point), `bottomRight` (Point), `i` (int): index. |
| `lines`         | `list`                     | List of lines in the figure, each having: `start` (Point): Coordinates of start point, `end` (Point): Endpoint coordinates, `i` (int): index. |
| `beziers`       | `list`                     | List of Bezier curves, each having: `start` (Point): Start point, `p1` (Point): Control point 1, `p2` (Point): Control point 2, `end` (Point): End point, `i` (int): index. |
| `boundingBox`   | `list`                     | The bounding box around the entire figure, with: `[x1,y1,x2,y2]` (float,float,float,float) . |
| `caption`       | `str`                      | Text describing the figure, extracted from the document. |
| `img`           | `str` (base64-encoded)     | Base64-encoded string of the figure image for easy embedding. |
| `hog_id`       | `int`              | HouseofGraph id if graph exists in the database, else `-1` |

---
Points are dictionaries with `x` and `y` keys, e.g., `{"x": 0.0, "y": 0.0}`.

## Initialization

The `app.py` file initializes the FastAPI application and sets up the necessary middleware and routes.

### CORS Middleware

The CORS middleware is configured to allow requests from specified origins. The allowed origins are set using the `CORS_ALLOW_ORIGINS` environment variable.