# SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
# SPDX-FileCopyrightText: 2024 Tim Hegemann <tim.hegemann@uni-wuerzburg.de>
# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later


import concurrent.futures
import json
import warnings
from functools import partial
from typing import Any, List, Optional

import networkx as nx
import requests
from geometry_objects import Circle, Edge, Rect
from geometry_util import (
    create_matrix,
    detect_circles_from_beziers,
    detect_edges_from_beziers,
    detect_edges_from_lines,
    extract_gemetric_elements,
    filter_duplicate_vertices,
    filter_vertices_based_on_peak_size,
    find_connected_components,
    max_node_degree,
)
from nptyping import NDArray
from nptyping.typing_ import Bool


def process_df(file_path: str):
    from kieta_modules.pipeline import PipelineManager

    # Flask
    pipelineManager: PipelineManager = PipelineManager()
    pipelineManager.read_from_file("backend/pipeline.json")

    with open(file_path, "rb") as doc:
        doc = {
            "file": doc.read(),
            "id": "".join(file_path.split(".")[:-1]),
            "suffix": file_path.split(".")[-1],
        }

    res = pipelineManager.get_pipeline("GraphExtraction").process_full(doc)

    return res


def get_hog_id(graph: NDArray[Any, Bool]) -> Optional[int]:
    url = "https://houseofgraphs.org/api/enquiry"

    graph6 = nx.to_graph6_bytes(nx.from_numpy_array(graph)).decode("ascii").strip()

    payload = {
        "canonicalFormEnquiry": {"canonicalForm": graph6},
        "formulaEnquiries": [],
        "graphClassEnquiries": [],
        "interestingInvariantEnquiries": [],
        "invariantEnquiries": [],
        "invariantParityEnquiries": [],
        "invariantRangeEnquiries": [],
        "mostPopular": -1,
        "mostRecent": -1,
        "subgraphEnquiries": [],
        "textEnquiries": [],
    }

    # Convert the payload to JSON format
    json_payload = json.dumps(payload)

    # Set the headers for the POST request
    headers = {"Content-Type": "application/json"}

    # Make the POST request with a timeout of 20 seconds
    try:
        response = requests.post(url, data=json_payload, headers=headers, timeout=20)
    except requests.exceptions.Timeout:
        raise Exception("Request to HoG timed out")

    try:
        if response.json()["totalCandidates"] > 1:
            raise Exception("WARNING: More than one candidate returned")

        if response.json()["totalCandidates"] == 0:
            return None

        return response.json()["_embedded"]["graphSearchModelList"][0]["graphId"]
    except (requests.exceptions.RequestException, KeyError, ValueError):
        raise Exception("Error in HoG response: " + response.text)


def process_figure(figure):
    lines, rects, quads, beziers, _ = extract_gemetric_elements(
        figure["objs"], ignore_rect_without_color=True
    )

    if len(rects) == 0 and len(quads) == 0:
        _, rects, quads, _, _ = extract_gemetric_elements(
            figure["objs"], ignore_rect_without_color=False
        )

    circles, beziers = detect_circles_from_beziers(beziers)

    edges: List[Edge] = []

    vertex_candidates = circles + rects + quads

    vertex_candidates, regained_beziers, regained_lines = (
        filter_vertices_based_on_peak_size(vertex_candidates, beziers, lines)
    )
    beziers += regained_beziers
    lines += regained_lines

    vertex_candidates = filter_duplicate_vertices(vertex_candidates)

    vertex_candidates.sort(key=lambda vertex: vertex.center[0])

    if len(vertex_candidates) <= 6:
        return None

    edges = detect_edges_from_lines(vertex_candidates, lines, edges)

    edges = detect_edges_from_beziers(vertex_candidates, beziers, lines, edges)

    subgraphs_vertices = find_connected_components(vertex_candidates, edges)

    subgraphs_matrices = [
        g
        for subgraph_vertices in subgraphs_vertices
        if (g := create_matrix(subgraph_vertices, edges)).shape[0] > 6
        and max_node_degree(g) > 2
    ]

    graph6_strings = [
        nx.to_graph6_bytes(nx.from_numpy_array(graph)).decode("ascii").strip()
        for graph in subgraphs_matrices
    ]
    hog_ids = [
        t if (t := get_hog_id(graph)) is not None else -1
        for graph in subgraphs_matrices
    ]

    subgraph_circles = []
    subgraph_rects = []
    for subgraph_vertices in subgraphs_vertices:
        indexed_circles = []
        indexed_rects = []
        for i, vertex in enumerate(subgraph_vertices):
            vertex.index = i
            if isinstance(vertex, Circle):
                indexed_circles.append(vertex)
            elif isinstance(vertex, Rect):
                indexed_rects.append(vertex)
        subgraph_circles.append(indexed_circles)
        subgraph_rects.append(indexed_rects)

    figure_json = []
    for i in range(len(graph6_strings)):
        figure_json.append(
            {
                "graph6_strings": graph6_strings[i],
                "circles": [
                    circle.__dict__() for circle in subgraph_circles[i] if circle.used
                ],
                "rects": [rect.__dict__() for rect in subgraph_rects[i] if rect.used],
                "lines": [line.__dict__() for line in lines if line.used],
                "beziers": [bezier.__dict__() for bezier in beziers if bezier.used],
                "boundingBox": figure["boundingBox"],
                "caption": figure["caption"],
                "img": figure["img"],
                "hog_id": hog_ids[i],
            }
        )

    # filter empties
    figure_json = [figure for figure in figure_json if figure["graph6_strings"]]

    return figure_json


def harvest_graph(file_path):
    data = None
    if isinstance(file_path, str):
        if file_path[-3:] == "pdf":
            data = process_df(file_path)
        elif file_path[-4:] == "json":
            with open(file_path) as f:
                data = json.load(f)
    elif isinstance(file_path, dict) or isinstance(file_path, list):
        data = file_path
    else:
        print(f"Unsupported file type: {type(file_path)}")

    if data is None:
        return None

    warnings.filterwarnings("ignore")

    data.sort(key=lambda x: (x["page"], x["boundingBox"][1]))

    json_output = []

    process_figure_partial = partial(process_figure)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_figure = {
            executor.submit(process_figure_partial, figure): figure for figure in data
        }

        for future in concurrent.futures.as_completed(future_to_figure):
            figure = future_to_figure[future]
            try:
                result = future.result()
                if result is not None:
                    json_output.append(result)
            except Exception as exc:
                print(f'Figure {figure["caption"]} generated an exception: {exc}')

        # remove redundant arrays in json_output
        json_output = [
            {k: v for element in figure for k, v in element.items()}
            for figure in json_output
            if figure
        ]

        return json_output
