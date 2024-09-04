import json
import numpy as np
from nptyping import NDArray
from nptyping.typing_ import Bool
import sys
from typing import List, Optional, Any
import argparse
import requests
import networkx as nx
import warnings
from geometry_objects import *
from geometry_util import *


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

    # Make the POST request
    try:
        response = requests.post(url, data=json_payload, headers=headers)
    except requests.exceptions.Timeout:
        raise Exception("Request to HoG timed out")

    try:
        if response.json()["totalCandidates"] > 1:
            raise Exception("WARNING: More than one candidate returned")

        if response.json()["totalCandidates"] == 0:
            return None

        return response.json()["_embedded"]["graphSearchModelList"][0]["graphId"]
    except:
        raise Exception("Error in HoG response: " + response.text)


def harvest_graph(file_path, gui=False, server_mode=True):
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

    if gui:
        from drawing_util import Drawer

        drawer = Drawer()

    warnings.filterwarnings("ignore")

    np.set_printoptions(threshold=sys.maxsize)

    first = True
    running = True
    while running:
        if gui:
            if drawer.check_for_quit():
                running = False
                break

        num_figure = 0
        total_num_graphs = 0
        total_num_interesting_graphs = 0
        total_num_interesting_graphs_in_hog = 0

        # empty json
        json_output = []
        for figure in data:
            num_figure += 1

            lines, rects, quads, beziers, labels = extract_gemetric_elements(
                figure["objs"]
            )

            circles, beziers = detect_circles_from_beziers(beziers)

            edges: List[Edge] = []

            rects, lines_from_rects = try_converting_rect_to_lines(rects, circles)
            lines += lines_from_rects

            quads, lines_from_rects = try_converting_rect_to_lines(quads, circles)
            lines += lines_from_rects

            mimic_rects_as_circles = lambda rect: Circle(
                rect.width / 2,
                (rect.topLeft[0] + rect.width / 2, rect.topLeft[1] + rect.height / 2),
                False,
            )

            circles += map(mimic_rects_as_circles, rects)

            old_size = len(circles)
            # Filter out circles that are irregular in size
            circles, regained_beziers = filter_circles_based_on_size(circles)

            beziers += regained_beziers

            # TODO: Sort circles and lines to enable binary search
            # def compareC(circle1, circle2):
            #     if circle1.center[0] == circle2.center[0]:
            #         if circle1.center[1] == circle2.center[1]:
            #             return circle1.radius - circle2.radius
            #         return circle1.center[1] - circle2.center[1]
            #     else:
            #         return circle1.center[0] - circle2.center[0]

            # circles.sort(key=cmp_to_key(compareC))

            # def compareL(line1, line2):
            #     if line1.start[0] == line2.start[0]:
            #         return line1.start[1] - line2.start[1]
            #     else:
            #         return line1.start[0] - line2.start[0]

            # lines.sort(key=cmp_to_key(compareL))

            # Filter out circles contained in other circles
            circles = filter_duplicate_circles(circles)

            if first:
                # print(
                #     "Number of verticies filtered out based on size:",
                #     old_size - middle_size,
                # )
                # print(
                #     "Number of verticies filtered out because of overlap:",
                #     middle_size - new_size,
                # )

                if len(circles) <= 6 and not gui:
                    total_num_graphs += 1
                    continue

                adjustence_matrix = np.zeros((len(circles), len(circles)), dtype=bool)

                # Detect edges from lines
                edges = detect_edges_from_lines(circles, lines, edges)

                edges = detect_edges_from_beziers(circles, beziers, lines, edges)

                adjustence_matrix = create_matrix(circles, edges)

                circles_copy = circles.copy()

                matrix_without_copies = create_matrix(circles_copy, edges)

                graphs = compute_subgraphs(matrix_without_copies)

                num_graphs = len(graphs)

                old_size = len(graphs)
                graphs = [graph for graph in graphs if graph.shape[0] > 6]

                graphs = [graph for graph in graphs if max_node_degree(graph) > 2]

                num_interesting_graphs = len(graphs)

                # print(
                #     "Number of graphs filtered out based on size:",
                #     old_size - len(graphs),
                # )

                if len(graphs) > 0 and gui and not server_mode:

                    print(
                        len(graphs),
                        "graphs containing",
                        np.sum([graph.shape[0] for graph in graphs]),
                        "vertices in figure",
                        # figures["caption"],
                        "detected. Index:",
                        num_figure,
                    )

                    print(
                        len(graphs),
                        "graphs exist in Figure",
                    )
                # NOTE: apparently not needed?
                # num_graphs_in_hog = sum(map(lambda x: get_hog_id(x) != None, graphs))

                graph6_strings = [
                    nx.to_graph6_bytes(nx.from_numpy_array(graph))
                    .decode("ascii")
                    .strip()
                    for graph in graphs
                ]
                hog_ids = [get_hog_id(graph) for graph in graphs]
                hog_ids = [hog_id if hog_id != None else -1 for hog_id in hog_ids]

                figure_json = [
                    {"graph6_strings": graph6_strings},
                    {
                        "circles": [
                            circle.__dict__() for circle in circles if circle.used
                        ]
                    },
                    {"lines": [line.__dict__() for line in lines if line.used]},
                    {
                        "beziers": [
                            bezier.__dict__() for bezier in beziers if bezier.used
                        ]
                    },
                    {"boundingBox": figure["boundingBox"]},
                    {"caption": figure["caption"]},
                    {"img": figure["img"]},
                    {"hog_ids": hog_ids},
                ]

                json_output.append(figure_json)

            if gui:
                drawer.draw_objects(circles, lines, rects, quads, beziers, labels)

        if first:
            if server_mode:
                # print(json_output)
                # print(json.dumps(json_output))

                # remove redundant arrays in json_output
                json_output = [
                    {k: v for element in figure for k, v in element.items()}
                    for figure in json_output
                ]

                return json_output
            else:
                print("Total number of graphs:", total_num_graphs)
                print(
                    "Total number of interesting graphs:", total_num_interesting_graphs
                )
                print(
                    "Total number of interesting graphs in House of Graphs:",
                    total_num_interesting_graphs_in_hog,
                )
            first = False
            if not gui or server_mode:
                running = False

        if gui:
            drawer.update()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph harvester")
    parser.add_argument(
        "--file", "-f", type=str, help="PDF/JSON to harvest graphs from", required=True
    )
    parser.add_argument(
        "--no-gui",
        dest="no_gui",
        action="store_false",
        help="Flag to not display a gui (optional)",
    )
    parser.add_argument(
        "--server-mode",
        dest="server_mode",
        action="store_true",
        help="Flag to switch to server mode (optional)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    harvest_graph(args.file, args.no_gui, args.server_mode)
