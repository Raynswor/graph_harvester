# SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later


import math
import sys
from typing import Any, List, Optional, Union

import numpy as np
from geometry_objects import Bezier, Circle, Edge, Label, Line, Rect, Vertex
from nptyping import NDArray
from nptyping.typing_ import Bool

# Percentage a line can be outside of a vertex to still be counted as connected
OVERLAPPING_TOLERANCE = 0.5

# Distance of endpoints in which two lines are considered connected
CONNECT_LINES_THRESHOLD = 0.2

# Percentage a radius can differ from the average radius of all circles
KEEP_VERTEX_THESHOLD = 0.3

# Distance at which a label can be assigned to a circle
# TODO: this should be a dynamic value/percentage
MAX_DISTANCE_FOR_LABEL = 15

# Max percentage the two radii of a circle (given by their PDF representation)
# can differ to still be counted as a circle
CIRCLE_THRESHOLD = 0.2


def extract_gemetric_elements(
    objs: dict, ignore_rect_without_color=True
) -> tuple[List[Line], List[Rect], List[Rect], List[Bezier], List[Label]]:
    lines: List[Line] = []
    rects: List[Rect] = []
    quads: List[Rect] = []
    beziers: List[Bezier] = []
    labels: List[Label] = []

    # logger = logging.getLogger("main")

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

            if ignore_rect_without_color and "color" not in obj["draw"].keys():
                # For some rects, this indicates their invisible
                # logger.warning("No color in DrawnRectangle")
                i += 1
                continue
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


def num_elements_starting_in_circle(
    circle: Circle, elements: List[Union[Line, Bezier]]
) -> int:
    count = 0
    for element in elements:
        if circle_contains_point(circle, element.start) and not circle_contains_point(
            circle, element.stop
        ):
            count += 1
        if circle_contains_point(circle, element.stop) and not circle_contains_point(
            circle, element.start
        ):
            count += 1
    return count


def num_elements_starting_in_rect(
    rect: Rect, elements: List[Union[Line, Bezier]]
) -> int:
    count = 0
    for element in elements:
        if rect_contains_point(rect, element.start) and not rect_contains_point(
            rect, element.stop
        ):
            count += 1
        if rect_contains_point(rect, element.stop) and not rect_contains_point(
            rect, element.start
        ):
            count += 1
    return count


def filter_vertices_based_on_peak_size(
    vertices: List[Vertex], lines: List[Line], beziers: List[Bezier]
) -> tuple[List[Vertex], List[Bezier], List[Line]]:
    if len(vertices) < 10:
        return filter_vertices_based_on_avg_size(vertices)

    min_count = 5

    area_groups = []

    for vertex in vertices:
        area_found = False
        for group in area_groups:
            average = np.mean([v.area for v in group])
            # Group vertices with similar radii (within tolerance)
            if abs(average - vertex.area) <= average * KEEP_VERTEX_THESHOLD:
                group.append(vertex)
                area_found = True
                break
        if not area_found:
            area_groups.append([vertex])

    # Filter groups with fewer than min_count circles
    filtered_vertices = []
    further_candidates = []
    regained_beziers = []
    regained_lines = []
    for group in area_groups:
        if len(group) < min_count:
            further_candidates.extend(group)
            continue

        centers_excluding = [
            vertex.center for vertex in vertices if vertex not in group
        ]
        avg_excluding = np.mean(
            [vertex.area for vertex in vertices if vertex not in group]
        )
        isBigger = np.mean([vertex.area for vertex in group]) / avg_excluding >= 3
        if isBigger:
            # Inefficient, but should not be rached often
            for vertex in group:
                if isinstance(vertex, Circle):
                    containsVertex = any(
                        circle_contains_point(vertex, center)
                        for center in centers_excluding
                    )
                    if not containsVertex:
                        filtered_vertices.append(vertex)
                    else:
                        regained_beziers.extend(vertex.original_beziers)
                else:
                    containsVertex = any(
                        rect_contains_point(vertex, center)
                        for center in centers_excluding
                    )
                    if not containsVertex:
                        filtered_vertices.append(vertex)
                    else:
                        regained_lines.extend(vertex.get_lines())
        else:
            filtered_vertices.extend(group)

    # Important vertices my be bigger and smaller in number
    for vertex in further_candidates:
        # Check if candidate contains any other circles and edge candidates start in it
        if isinstance(vertex, Circle):
            circle = vertex
            if (
                not any(
                    circle_contains_point(circle, filtered_vertex.center)
                    for filtered_vertex in filtered_vertices
                )
                and num_elements_starting_in_circle(circle, lines + beziers) >= 3
            ):
                filtered_vertices.append(circle)
            else:
                regained_beziers.extend(circle.original_beziers)
        else:
            rect = vertex
            if (
                not any(
                    rect_contains_point(rect, filtered_vertex.center)
                    for filtered_vertex in filtered_vertices
                )
                and num_elements_starting_in_rect(rect, lines + beziers) >= 3
            ):
                filtered_vertices.append(rect)
            else:
                regained_lines.extend(rect.get_lines())

    # Step 6: Return the filtered list of circles
    return filtered_vertices, regained_beziers, regained_lines


def filter_vertices_based_on_avg_size(
    vertices: List[Vertex],
) -> tuple[List[Vertex], List[Bezier], List[Line]]:
    average_area = np.mean([vertex.area for vertex in vertices])
    filtered_vertices = []
    regained_beziers = []
    regained_lines = []
    for vertex in vertices:
        if (
            vertex.area < (1 + KEEP_VERTEX_THESHOLD) * average_area
            and vertex.area > (1 - KEEP_VERTEX_THESHOLD) * average_area
        ):
            filtered_vertices.append(vertex)
        else:
            # If vertex is too big or too small, regain the components as they may be edges
            if isinstance(vertex, Circle):
                regained_beziers.extend(vertex.original_beziers)
            if isinstance(vertex, Rect):
                regained_lines.extend(vertex.get_lines())

    return filtered_vertices, regained_beziers, regained_lines


def circle_contains_point(circle: Circle, point: tuple[int, int]) -> bool:
    max_distance = circle.radius * (1 + OVERLAPPING_TOLERANCE)
    if (
        abs(circle.center[0] - point[0]) > max_distance
        or abs(circle.center[1] - point[1]) > max_distance
    ):
        return False

    return lays_within_distance(circle.center, point, max_distance)


def rect_contains_point(rect: Rect, point: tuple[int, int]) -> bool:
    max_width_distance_tolerance = rect.width * OVERLAPPING_TOLERANCE
    max_height_distance_tolerance = rect.height * OVERLAPPING_TOLERANCE

    return (
        rect.topLeft[0] - max_width_distance_tolerance
        <= point[0]
        <= rect.bottomRight[0] + max_width_distance_tolerance
        and rect.topLeft[1] - max_height_distance_tolerance
        <= point[1]
        <= rect.bottomRight[1] + max_height_distance_tolerance
    )


def filter_duplicate_vertices(vertices: List[Vertex]) -> List[Vertex]:
    to_remove = []
    vertices = list(dict.fromkeys(vertices))  # Remove duplicates while perserving order
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            vertex1 = vertices[i]
            vertex2 = vertices[j]
            if isinstance(vertex1, Circle) and isinstance(vertex2, Circle):
                if lays_within_distance(
                    vertex1.center, vertex2.center, vertex1.radius + vertex2.radius
                ):
                    if vertex1.radius >= vertex2.radius:
                        to_remove.append(vertex2)
                    else:
                        to_remove.append(vertex1)
            elif isinstance(vertex1, Rect) and isinstance(vertex2, Rect):
                if (
                    rect_contains_point(vertex1, vertex2.topLeft)
                    or rect_contains_point(vertex1, vertex2.bottomRight)
                    or rect_contains_point(
                        vertex1, (vertex2.topLeft[0], vertex2.bottomRight[1])
                    )
                    or rect_contains_point(
                        vertex1, (vertex2.bottomRight[0], vertex2.topLeft[1])
                    )
                ):
                    if vertex1.area >= vertex2.area:
                        to_remove.append(vertex2)
                    else:
                        to_remove.append(vertex1)
            else:
                circle = vertex1 if isinstance(vertex1, Circle) else vertex2
                rect = vertex1 if isinstance(vertex1, Rect) else vertex2
                if (
                    circle_contains_point(circle, rect.topLeft)
                    or circle_contains_point(circle, rect.bottomRight)
                    or circle_contains_point(
                        circle, (rect.topLeft[0], rect.bottomRight[1])
                    )
                    or circle_contains_point(
                        circle, (rect.bottomRight[0], rect.topLeft[1])
                    )
                ):
                    if vertex1.area >= vertex2.area:
                        to_remove.append(vertex2)
                    else:
                        to_remove.append(vertex1)

    return [vertex for vertex in vertices if vertex not in to_remove]


def get_vertex_containing_point(
    vertices: List[Vertex], point: tuple[int, int]
) -> Optional[Vertex]:
    # TODO: This can be optimized by using binary search
    for vertex in vertices:
        if isinstance(vertex, Circle) and circle_contains_point(vertex, point):
            return vertex
        if isinstance(vertex, Rect) and rect_contains_point(vertex, point):
            return vertex

    return None


def distance_point_to_line(point: tuple[int, int], line: Line) -> float:
    x0, y0 = point
    x1, y1 = line.start
    x2, y2 = line.stop

    # Calculate the components of the line equation: ax + by + c = 0 - Part 1
    a = y2 - y1

    # Check if the perpendicular projection of the point falls within the line segment
    dot_product = (x0 - x1) * (x2 - x1) + (y0 - y1) * a
    if dot_product < 0:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    squared_length = (x2 - x1) ** 2 + a**2
    if dot_product > squared_length:
        return math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)

    # Calculate the components of the line equation: ax + by + c = 0 - Part 2
    b = x1 - x2
    c = (x2 * y1) - (x1 * y2)
    # Calculate the distance using the formula: |ax0 + by0 + c| / sqrt(a^2 + b^2)
    return abs((a * x0) + (b * y0) + c) / math.sqrt((a**2) + (b**2))


def line_intersects_rectangle(x1, y1, x2, y2, rect_x, rect_y, rect_width, rect_height):
    ## liang-barsky algorithm
    t0, t1 = 0, 1
    dx = x2 - x1
    dy = y2 - y1

    # Check intersection with vertical edges
    p = [-dx, dx, -dy, dy]
    q = [x1 - rect_x, rect_x + rect_width - x1, y1 - rect_y, rect_y + rect_height - y1]

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False

    return True


def get_vertices_on_line(vertices: List[Vertex], line: Line) -> List[Vertex]:
    vertices_on_line = []
    distances_squared = []
    line_start = np.array(line.start)

    for vertex in vertices:
        if isinstance(vertex, Circle):
            circle = vertex
            center = np.array(circle.center)
            distance_to_line = distance_point_to_line(circle.center, line)
            if distance_to_line <= circle.radius:
                distance_start_squared = np.sum((center - line_start) ** 2)
                vertices_on_line.append(circle)
                distances_squared.append(distance_start_squared)
        if isinstance(vertex, Rect):
            rect = vertex
            if line_intersects_rectangle(
                *line.start, *line.stop, *rect.topLeft, rect.width, rect.height
            ):
                rect_center = np.array(
                    [
                        (rect.topLeft[0] + rect.bottomRight[0]) / 2,
                        (rect.topLeft[1] + rect.bottomRight[1]) / 2,
                    ]
                )
                distance_start_squared = np.sum(
                    (np.array(rect_center) - line_start) ** 2
                )
                vertices_on_line.append(rect)
                distances_squared.append(distance_start_squared)

    return [vertices_on_line[i] for i in np.argsort(distances_squared)]


def get_vertices_on_bezier(vertices: List[Vertex], bezier: Bezier) -> List[Vertex]:
    vertices_on_line = []
    for point in bezier.points:
        for vertex in vertices:
            if vertex in vertices_on_line:
                continue
            if isinstance(vertex, Circle):
                circle = vertex
                max_distance = circle.radius
                if (
                    abs(circle.center[0] - point[0]) > max_distance
                    or abs(circle.center[1] - point[1]) > max_distance
                ):
                    continue

                if (
                    lays_within_distance(circle.center, point, max_distance)
                    and circle not in vertices_on_line
                ):
                    vertices_on_line.append(circle)
            if isinstance(vertex, Rect):
                rect = vertex
                if rect_contains_point(rect, point) and rect not in vertices_on_line:
                    vertices_on_line.append(rect)

    return vertices_on_line


def detect_edges_from_lines(
    vertices: List[Vertex], lines: List[Line], edges: List[Edge] = []
) -> List[Edge]:
    for line in lines:
        begin = get_vertex_containing_point(vertices, line.start)
        path = [line]

        while not begin:
            new_begin, additionalLine = try_extending_point_by_line(
                path[0].start, [line for line in lines if line not in path]
            )
            if additionalLine:
                begin = get_vertex_containing_point(vertices, new_begin)
                path.insert(0, additionalLine)
            else:
                break

        if not begin:
            continue

        end = get_vertex_containing_point(vertices, line.stop)

        while not end:
            new_end, additionalLine = try_extending_point_by_line(
                path[-1].stop, [line for line in lines if line not in path]
            )
            end = get_vertex_containing_point(vertices, new_end)
            if additionalLine:
                path.append(additionalLine)
            else:
                break

        if begin is not None and end is not None and begin != end:
            # If the line was not extended, include all other verticies on the line as well
            laying_on_line = []
            for component in path:
                if isinstance(component, Line):
                    laying_on_line.extend(get_vertices_on_line(vertices, component))
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
    vertices: List[Vertex],
    beziers: List[Bezier],
    lines: List[Line],
    edges: List[Edge],
) -> List[Edge]:
    for bezier in beziers:
        path = [bezier]  # List of beziers/lines that connect begin to end
        begin = get_vertex_containing_point(vertices, bezier.start)

        while not begin:
            new_begin, additionalBezier = try_extending_point_by_bezier(
                path[0].start, [bezier for bezier in beziers if bezier not in path]
            )
            if additionalBezier:
                begin = get_vertex_containing_point(vertices, new_begin)
                path.insert(0, additionalBezier)

            if not begin:
                new_begin, additionalLine = try_extending_point_by_line(
                    path[0].start, [line for line in lines if line not in path]
                )
                if additionalLine:
                    begin = get_vertex_containing_point(vertices, new_begin)
                    path.insert(0, additionalLine)
                else:
                    break

        # Bezier does not start in a circle, so continue
        if begin is None:
            continue

        end = get_vertex_containing_point(vertices, bezier.stop)

        while not end:
            new_end, additionalBezier = try_extending_point_by_bezier(
                path[-1].stop, [bezier for bezier in beziers if bezier not in path]
            )
            if additionalBezier:
                end = get_vertex_containing_point(vertices, new_end)
                path.append(additionalBezier)
                continue
            if not end:
                new_end, additionalLine = try_extending_point_by_line(
                    path[-1].stop, [line for line in lines if line not in path]
                )
                if additionalLine:
                    end = get_vertex_containing_point(vertices, new_end)
                    path.append(additionalLine)
                else:
                    break

        if end is not None and begin != end:
            vertices_on_path = []
            for component in path:
                if isinstance(component, Bezier):
                    vertices_on_path.extend(get_vertices_on_bezier(vertices, component))
                if isinstance(component, Line):
                    vertices_on_path.extend(get_vertices_on_line(vertices, component))
                component.used = True

            if len(vertices_on_path) == 0 or vertices_on_path[0] != begin:
                vertices_on_path.insert(0, begin)
            if vertices_on_path[-1] != end:
                vertices_on_path.append(end)

            for i in range(len(vertices_on_path) - 1):
                edge1 = Edge(vertices_on_path[i], vertices_on_path[i + 1])
                edge2 = Edge(vertices_on_path[i + 1], vertices_on_path[i])
                if edge1 not in edges:
                    edges.append(edge1)
                    edges.append(edge2)
                vertices_on_path[i].used = True
                vertices_on_path[i + 1].used = True

    return edges


def create_matrix(circles: List[Vertex], edges: List[Edge]) -> NDArray[Any, Bool]:
    matrix = np.zeros((len(circles), len(circles)), dtype=bool)
    for edge in edges:
        if edge.start in circles and edge.stop in circles:
            matrix[circles.index(edge.start)][circles.index(edge.stop)] = 1

    # remove empty rows and columns
    matrix = matrix[:, ~np.all(matrix == 0, axis=0)]
    matrix = matrix[~np.all(matrix == 0, axis=1)]
    return matrix


def dfs(start: Vertex, edges: List[Edge], visited: set) -> list[Vertex]:
    stack = [start]
    connected_component = []

    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue

        visited.add(vertex)
        connected_component.append(vertex)

        for edge in edges:
            if edge.start == vertex:
                stack.append(edge.stop)
            if edge.stop == vertex:
                stack.append(edge.start)

    return connected_component


def find_connected_components(
    vertices: List[Vertex], edges: List[Edge]
) -> list[list[Vertex]]:
    visited = set()
    connected_components = []

    for vertex in vertices:
        if vertex not in visited:
            component = dfs(vertex, edges, visited)
            connected_components.append(component)

    return connected_components


def max_node_degree(adj_matrix: NDArray[Any, Bool]) -> int:
    if adj_matrix.shape[0] == 0:
        return 0

    max_sum = np.max(np.sum(adj_matrix, axis=0))
    return min(max_sum, adj_matrix.shape[0] - 1)
