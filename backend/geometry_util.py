from typing import List, Optional, Any
import numpy as np
import sys
import math
from nptyping import NDArray
from nptyping.typing_ import Bool

from geometry_objects import *


# Percentage a line can be outside of a vertex to still be counted as connected
OVERLAPPING_TOLERANCE = 0.5

# Distance of endpoints in which two lines are considered connected
CONNECT_LINES_THRESHOLD = 0.2

# Percentage a radius can differ from the average radius of all circles
KEEP_VERTEX_THESHOLD = 0.3

# Distance at which a label can be assigned to a circle
# TODO: this should be a dynamic value/percentage
MAX_DISTANCE_FOR_LABEL = 15

# Max percentage the two diameters of a circle (given by their PDF representation)
# can differ to still be counted as a circle
CIRCLE_THRESHOLD = 0.2


def extract_gemetric_elements(
    objs: dict,
) -> tuple[List[Line], List[Rect], List[Rect], List[Bezier], List[Label]]:
    lines: List[Line] = []
    rects: List[Rect] = []
    quads: List[Rect] = []
    beziers: List[Bezier] = []
    labels: List[Label] = []

    unknown = set(
        filter(
            lambda x: x
            not in [
                "DrawnRectangle",
                "String",
                "DrawnLine",
                "DrawnBezier",
                "DrawnQuad",
            ],
            map(lambda x: x["type"], objs),
        )
    )
    if len(unknown) > 0:
        print("UNKNOWN OBJECTS:", unknown, file=sys.stderr)

    objs = list(
        filter(
            lambda x: x["type"] in ["DrawnRectangle", "String", "DrawnQuad"]
            or x["type"] in ["DrawnLine", "DrawnBezier"]
            and "pts" in x.keys(),
            objs,
        )
    )

    i = 0
    while i < len(objs):
        obj = objs[i]
        if obj["type"] == "DrawnLine":
            points = obj["pts"]
            lines.append(Line(points[0], points[1]))
        elif obj["type"] == "DrawnRectangle":
            width = obj["pts"][2] - obj["pts"][0]
            height = obj["pts"][3] - obj["pts"][1]
            filled = obj["draw"]["type"] == "f"
            # If width and height differ too much, it is not a rectangle e.g. not a vertex
            if (
                width == 0
                or height == 0
                or width / height > (1 + CIRCLE_THRESHOLD)
                or height / width > (1 + CIRCLE_THRESHOLD)
            ):
                i += 1
                quads.append(Rect(obj["pts"][0:2], obj["pts"][2:4], filled))
                continue

            rects.append(Rect(obj["pts"][0:2], obj["pts"][2:4], filled))
        elif obj["type"] == "DrawnQuad":
            quads.append(Rect(obj["pts"][0:2], obj["pts"][2:4], filled))
        elif obj["type"] == "DrawnBezier":
            points = obj["pts"]

            beziers.append(Bezier(points[0], points[1], points[2], points[3]))
        elif obj["type"] == "String":
            labels.append(
                Label(
                    obj["boundingBox"][0:2],
                    (
                        obj["boundingBox"][2] - obj["boundingBox"][0],
                        obj["boundingBox"][3] - obj["boundingBox"][1],
                    ),
                    obj["content"],
                )
            )

        else:
            pass
        # print("Unknown object type: " + obj["type"])
        # break

        i += 1

    return lines, rects, quads, beziers, labels


def detect_circles_from_beziers(
    beziers: List[Bezier],
) -> tuple[List[Circle], List[Bezier]]:
    circles = []
    used_for_circle = []
    for i in range(len(beziers)):
        j = i
        while j + 1 < len(beziers):
            if not np.allclose(beziers[j].stop, beziers[j + 1].start):
                break
            j += 1

        if not np.allclose(beziers[j].stop, beziers[i].start):
            continue

        number_curves = j - i + 1

        if number_curves < 3:
            continue

        curves = beziers[i : j + 1]
        starting_points = [curve.start for curve in curves]

        center = np.mean(starting_points, axis=0)
        distances = [np.linalg.norm(point - center) for point in starting_points]
        radius = np.mean(distances)

        if any([distance > radius * (1 + CIRCLE_THRESHOLD) for distance in distances]):
            continue

        circles.append(Circle(radius, tuple(center.tolist()), True, curves))
        used_for_circle.extend(curves)
        continue

    return circles, [bezier for bezier in beziers if bezier not in used_for_circle]


def lays_within_distance(
    point1: tuple[float, float], point2: tuple[float, float], distance: float
) -> bool:
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 <= distance**2


def filter_circles_based_on_size(
    circles: List[Circle],
) -> tuple[List[Circle], List[Bezier]]:
    average_radius = np.mean([circle.radius for circle in circles])
    filtered_circles = []
    regained_beziers = []
    for circle in circles:
        if (
            circle.radius < (1 + KEEP_VERTEX_THESHOLD) * average_radius
            and circle.radius > (1 - KEEP_VERTEX_THESHOLD) * average_radius
        ):
            filtered_circles.append(circle)
        else:
            # If circle is too big or too small, regain the beziers as they may be edges
            regained_beziers.extend(circle.original_beziers)

    return filtered_circles, regained_beziers


def filter_duplicate_circles(circles: List[Circle]) -> List[Circle]:
    to_remove = []
    circles = list(dict.fromkeys(circles))  # Remove duplicates while perserving order
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            circle1 = circles[i]
            circle2 = circles[j]
            if abs(circle1.center[0] - circle2.center[0]) > (
                circle1.radius + circle2.radius
            ) or abs(circle1.center[1] - circle2.center[1]) > (
                circle1.radius + circle2.radius
            ):
                continue

            if lays_within_distance(
                circle1.center, circle2.center, circle1.radius + circle2.radius
            ):
                if circle1.radius >= circle2.radius:
                    to_remove.append(circle2)
                else:
                    to_remove.append(circle1)
    return [circle for circle in circles if circle not in to_remove]


def circle_containing_point(
    circles: List[Circle], point: tuple[int, int]
) -> Optional[Circle]:
    # TODO: This can be optimized by using binary search
    for circle in circles:
        max_distance = circle.radius * (1 + OVERLAPPING_TOLERANCE)
        if (
            abs(circle.center[0] - point[0]) > max_distance
            or abs(circle.center[1] - point[1]) > max_distance
        ):
            continue

        if lays_within_distance(circle.center, point, max_distance):
            return circle

    return None


def distance_circle_to_line(circle: Circle, line: Line) -> float:
    x0, y0 = circle.center
    x1, y1 = line.start
    x2, y2 = line.stop

    # Calculate the components of the line equation: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = (x2 * y1) - (x1 * y2)

    # Calculate the distance using the formula: |ax0 + by0 + c| / sqrt(a^2 + b^2)
    distance = abs((a * x0) + (b * y0) + c) / math.sqrt((a**2) + (b**2))

    # Check if the perpendicular projection of the point falls within the line segment
    dot_product = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
    if dot_product < 0:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if dot_product > squared_length:
        return math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)

    return distance


def circles_on_line(circles: List[Circle], line: Line) -> List[Circle]:
    circles_on_line = []
    distances_squared = []
    line_start = np.array(line.start)
    for circle in circles:
        center = np.array(circle.center)
        distance_to_line = distance_circle_to_line(circle, line)
        if distance_to_line <= circle.radius * (1 + OVERLAPPING_TOLERANCE):
            distance_start_squared = np.sum((center - line_start) ** 2)
            circles_on_line.append(circle)
            distances_squared.append(distance_start_squared)

    sorted_indices = np.argsort(distances_squared)

    return [circles_on_line[i] for i in sorted_indices]


def circles_on_bezier(circles: List[Circle], bezier: Bezier) -> List[Circle]:
    circles_on_line = []
    for point in bezier.points:
        j = 0
        for circle in circles:
            j += 1
            max_distance = circle.radius * (1 + OVERLAPPING_TOLERANCE)
            if (
                abs(circle.center[0] - point[0]) > max_distance
                or abs(circle.center[1] - point[1]) > max_distance
            ):
                continue

            if (
                lays_within_distance(circle.center, point, max_distance)
                and circle not in circles_on_line
            ):
                circles_on_line.append(circle)

    return circles_on_line


def detect_edges_from_lines(
    circles: List[Circle], lines: List[Line], edges: List[Edge] = []
) -> List[Edge]:
    for line in lines:
        begin = circle_containing_point(circles, line.start)
        path = [line]

        while not begin:
            new_begin, additionalLine = try_extending_point_by_line(
                path[0].start, [line for line in lines if line not in path]
            )
            if additionalLine:
                begin = circle_containing_point(circles, new_begin)
                path.insert(0, additionalLine)
            else:
                break

        if not begin:
            continue

        end = circle_containing_point(circles, line.stop)

        while not end:
            new_end, additionalLine = try_extending_point_by_line(
                path[-1].stop, [line for line in lines if line not in path]
            )
            end = circle_containing_point(circles, new_end)
            if additionalLine:
                path.append(additionalLine)
            else:
                break

        if begin is not None and end is not None and begin != end:
            # If the line was not extended, include all other verticies on the line as well
            laying_on_line = []
            for component in path:
                if isinstance(component, Line):
                    laying_on_line.extend(circles_on_line(circles, component))
                component.used = True

            if len(laying_on_line) == 0 or laying_on_line[0] != begin:
                laying_on_line.insert(0, begin)
            if laying_on_line[-1] != end:
                laying_on_line.append(end)

            for i in range(len(laying_on_line) - 1):
                edge1 = Edge(laying_on_line[i], laying_on_line[i + 1])
                edge2 = Edge(laying_on_line[i + 1], laying_on_line[i])
                if edge1 not in edges:
                    edges.append(edge1)
                    edges.append(edge2)
                laying_on_line[i].used = True
                laying_on_line[i + 1].used = True

    return edges


def try_extending_point_by_line(
    point: tuple[int, int], lines: List[Line]
) -> tuple[tuple[int, int], Optional[Line]]:
    for line in lines:
        if (
            abs(line.start[0] - point[0]) < CONNECT_LINES_THRESHOLD
            and abs(line.start[1] - point[1]) < CONNECT_LINES_THRESHOLD
        ):
            if lays_within_distance(line.start, point, CONNECT_LINES_THRESHOLD):
                return line.stop, line

        if (
            abs(line.stop[0] - point[0]) < CONNECT_LINES_THRESHOLD
            and abs(line.stop[1] - point[1]) < CONNECT_LINES_THRESHOLD
        ):
            if lays_within_distance(line.stop, point, CONNECT_LINES_THRESHOLD):
                return line.start, line

    return point, None


def try_extending_point_by_bezier(
    point: tuple[int, int], beziers: List[Bezier]
) -> tuple[tuple[int, int], Optional[Bezier]]:
    for bezier in beziers:
        if (
            abs(bezier.start[0] - point[0]) < CONNECT_LINES_THRESHOLD
            and abs(bezier.start[1] - point[1]) < CONNECT_LINES_THRESHOLD
        ):
            if lays_within_distance(bezier.start, point, CONNECT_LINES_THRESHOLD):
                return bezier.stop, bezier

        if (
            abs(bezier.stop[0] - point[0]) < CONNECT_LINES_THRESHOLD
            and abs(bezier.stop[1] - point[1]) < CONNECT_LINES_THRESHOLD
        ):
            if lays_within_distance(bezier.stop, point, CONNECT_LINES_THRESHOLD):
                return bezier.start, bezier

    return point, None


def detect_edges_from_beziers(
    circles: List[Circle],
    beziers: List[Bezier],
    lines: List[Line],
    edges: List[Edge],
) -> List[Edge]:
    for bezier in beziers:
        k = beziers.index(bezier)
        path = [bezier]  # List of beziers/lines that connect begin to end
        begin = circle_containing_point(circles, bezier.start)

        while not begin:
            new_begin, additionalBezier = try_extending_point_by_bezier(
                path[0].start, [bezier for bezier in beziers if bezier not in path]
            )
            if additionalBezier:
                begin = circle_containing_point(circles, new_begin)
                path.insert(0, additionalBezier)

            if not begin:
                new_begin, additionalLine = try_extending_point_by_line(
                    path[0].start, [line for line in lines if line not in path]
                )
                if additionalLine:
                    begin = circle_containing_point(circles, new_begin)
                    path.insert(0, additionalLine)
                else:
                    break

        # Bezier does not start in a circle, so continue
        if begin is None:
            continue

        end = circle_containing_point(circles, bezier.stop)

        while not end:
            new_end, additionalBezier = try_extending_point_by_bezier(
                path[-1].stop, [bezier for bezier in beziers if bezier not in path]
            )
            if additionalBezier:
                end = circle_containing_point(circles, new_end)
                path.append(additionalBezier)
                continue
            if not end:
                new_end, additionalLine = try_extending_point_by_line(
                    path[-1].stop, [line for line in lines if line not in path]
                )
                if additionalLine:
                    end = circle_containing_point(circles, new_end)
                    path.append(additionalLine)
                else:
                    break

        if end is not None and begin != end:
            circles_on_path = []
            for component in path:
                if isinstance(component, Bezier):
                    circles_on_path.extend(circles_on_bezier(circles, component))
                if isinstance(component, Line):
                    circles_on_path.extend(circles_on_line(circles, component))
                component.used = True

            if len(circles_on_path) == 0 or circles_on_path[0] != begin:
                circles_on_path.insert(0, begin)
            if circles_on_path[-1] != end:
                circles_on_path.append(end)

            for i in range(len(circles_on_path) - 1):
                edge1 = Edge(circles_on_path[i], circles_on_path[i + 1])
                edge2 = Edge(circles_on_path[i + 1], circles_on_path[i])
                if edge1 not in edges:
                    edges.append(edge1)
                    edges.append(edge2)
                circles_on_path[i].used = True
                circles_on_path[i + 1].used = True

    return edges


def try_converting_rect_to_lines(
    rects: List[Rect], circles: List[Circle]
) -> tuple[List[Rect], List[Line]]:
    toRemove = []
    toReturn = []
    for rect in rects:
        # ignore filled rectangles
        if rect.filled:
            continue

        if (
            circle_containing_point(circles, rect.topLeft) is not None
            and circle_containing_point(circles, (rect.topLeft[0], rect.bottomRight[1]))
            is not None
            and circle_containing_point(circles, (rect.bottomRight[0], rect.topLeft[1]))
            is not None
            and circle_containing_point(circles, rect.bottomRight) is not None
        ):
            toReturn.append(Line(rect.topLeft, (rect.topLeft[0], rect.bottomRight[1])))
            toReturn.append(Line(rect.topLeft, (rect.bottomRight[0], rect.topLeft[1])))
            toReturn.append(
                Line((rect.topLeft[0], rect.bottomRight[1]), rect.bottomRight)
            )
            toReturn.append(
                Line((rect.bottomRight[0], rect.topLeft[1]), rect.bottomRight)
            )
            toRemove.append(rect)
    rect = [rect for rect in rects if rect not in toRemove]
    return rect, toReturn


def create_matrix(circles: List[Circle], edges: List[Edge]) -> NDArray[Any, Bool]:
    matrix = np.zeros((len(circles), len(circles)), dtype=bool)
    for edge in edges:
        matrix[circles.index(edge.start)][circles.index(edge.stop)] = 1

    # remove empty rows and columns
    matrix = matrix[:, ~np.all(matrix == 0, axis=0)]
    matrix = matrix[~np.all(matrix == 0, axis=1)]
    return matrix


def find_connected_components(adj_matrix):
    n = adj_matrix.shape[0]
    visited = [False] * n
    components = []

    def dfs(node, component):
        component.append(node)
        visited[node] = True
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, component)

    for node in range(n):
        if not visited[node]:
            component = []
            dfs(node, component)
            components.append(component)

    return components


def extract_subgraph(adj_matrix, nodes):
    subgraph_size = len(nodes)
    subgraph = np.zeros((subgraph_size, subgraph_size), dtype=int)
    for i in range(subgraph_size):
        for j in range(subgraph_size):
            subgraph[i][j] = adj_matrix[nodes[i]][nodes[j]]
    return subgraph


def compute_subgraphs(adj_matrix: NDArray[Any, Bool]) -> List[NDArray[Any, Bool]]:
    components = find_connected_components(adj_matrix)
    if len(components) == 1:
        return [adj_matrix]
    subgraphs = []
    for component in components:
        subgraph = extract_subgraph(adj_matrix, component)
        subgraphs.append(subgraph)
    return subgraphs


def max_node_degree(adj_matrix: NDArray[Any, Bool]) -> int:
    if adj_matrix.shape[0] == 0:
        return 0

    max_sum = np.max(np.sum(adj_matrix, axis=0))
    return min(max_sum, adj_matrix.shape[0] - 1)
