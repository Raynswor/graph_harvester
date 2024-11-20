# SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later


import math
from typing import List, Union


# Taken from http://www.pygame.org/wiki/BezierCurve
def compute_bezier_points(vertices, numPoints=None) -> List[tuple[int, int]]:
    if numPoints is None:
        numPoints = 30
    if numPoints < 2 or len(vertices) != 4:
        return []

    result = []

    b0x = vertices[0][0]
    b0y = vertices[0][1]
    b1x = vertices[1][0]
    b1y = vertices[1][1]
    b2x = vertices[2][0]
    b2y = vertices[2][1]
    b3x = vertices[3][0]
    b3y = vertices[3][1]

    # Compute polynomial coefficients from Bezier points
    ax = -b0x + 3 * b1x + -3 * b2x + b3x
    ay = -b0y + 3 * b1y + -3 * b2y + b3y

    bx = 3 * b0x + -6 * b1x + 3 * b2x
    by = 3 * b0y + -6 * b1y + 3 * b2y

    cx = -3 * b0x + 3 * b1x
    cy = -3 * b0y + 3 * b1y

    dx = b0x
    dy = b0y

    # Set up the number of steps and step size
    numSteps = numPoints - 1  # arbitrary choice
    h = 1.0 / numSteps  # compute our step size

    # Compute forward differences from Bezier points and "h"
    pointX = dx
    pointY = dy

    firstFDX = ax * (h * h * h) + bx * (h * h) + cx * h
    firstFDY = ay * (h * h * h) + by * (h * h) + cy * h

    secondFDX = 6 * ax * (h * h * h) + 2 * bx * (h * h)
    secondFDY = 6 * ay * (h * h * h) + 2 * by * (h * h)

    thirdFDX = 6 * ax * (h * h * h)
    thirdFDY = 6 * ay * (h * h * h)

    # Compute points at each step
    result.append((int(pointX), int(pointY)))

    for i in range(numSteps):
        pointX += firstFDX
        pointY += firstFDY

        firstFDX += secondFDX
        firstFDY += secondFDY

        secondFDX += thirdFDX
        secondFDY += thirdFDY

        result.append((int(pointX), int(pointY)))

    return result


def toDict(point):
    return {"x": point[0], "y": point[1]}


class Circle:
    def __init__(self, radius, center, draw=True, original_beziers=[]) -> None:
        self.radius = radius
        self.center = center
        self.draw = draw
        self.used = False
        self.original_beziers = original_beziers
        self.area = self.radius**2 * math.pi
        self.index = None

        self.x = center[0]
        self.y = center[1]

    def __eq__(self, __value) -> bool:
        if not isinstance(__value, Circle):
            return False
        return self.radius == __value.radius and self.center == __value.center

    def __str__(self) -> str:
        return f"Circle(radius={self.radius}, center={self.center})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.radius, self.center))

    def __dict__(self):
        return {
            "r": self.radius,
            "c": toDict(self.center),
            "i": self.index,
        }


class Line:
    def __init__(self, start, stop) -> None:
        self.start = start
        self.stop = stop
        self.used = False

    def __eq__(self, value) -> bool:
        return self.start == value.start and self.stop == value.stop

    def __str__(self) -> str:
        return f"Line(start={self.start}, stop={self.stop})"

    def __repr__(self) -> str:
        return str(self)

    def __dict__(self):
        return {
            "start": toDict(self.start),
            "end": toDict(self.stop),
        }


class Rect:
    def __init__(self, topLeft, bottomRight, filled=True) -> None:
        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.width = bottomRight[0] - topLeft[0]
        self.height = bottomRight[1] - topLeft[1]
        self.filled = filled
        self.used = False
        self.area = self.width * self.height
        self.center = (topLeft[0] + self.width / 2, topLeft[1] + self.height / 2)
        self.index = None

        self.x1 = topLeft[0]
        self.y1 = topLeft[1]
        self.x2 = bottomRight[0]
        self.y2 = bottomRight[1]

    def __str__(self) -> str:
        return f"Rect(topLeft={self.topLeft}, bottomRight={self.bottomRight})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Rect):
            return False
        return self.topLeft == value.topLeft and self.bottomRight == value.bottomRight

    def __hash__(self) -> int:
        return hash((self.topLeft, self.bottomRight))

    def __dict__(self):
        return {
            "topLeft": toDict(self.topLeft),
            "bottomRight": toDict(self.bottomRight),
            "i": self.index,
        }

    def get_lines(self):
        return [
            Line(self.topLeft, (self.bottomRight[0], self.topLeft[1])),
            Line(self.topLeft, (self.topLeft[0], self.bottomRight[1])),
            Line((self.bottomRight[0], self.topLeft[1]), self.bottomRight),
            Line((self.topLeft[0], self.bottomRight[1]), self.bottomRight),
        ]


class Bezier:
    def __init__(self, start, p1, p2, stop) -> None:
        self.start = start
        self.p1 = p1
        self.p2 = p2
        self.stop = stop
        self.points = compute_bezier_points([start, p1, p2, stop], 100)
        self.used = False

    def __str__(self) -> str:
        return f"Bezier(start={self.start}, stop={self.stop})"

    def __repr__(self) -> str:
        return str(self)

    def __dict__(self):
        return {
            "start": toDict(self.start),
            "p1": toDict(self.p1),
            "p2": toDict(self.p2),
            "end": toDict(self.stop),
        }


Vertex = Union[Circle, Rect]


class Edge:
    def __init__(self, start: Vertex, stop: Vertex) -> None:
        self.start = start
        self.stop = stop

    def __eq__(self, value) -> bool:
        return self.start == value.start and self.stop == value.stop

    def __str__(self) -> str:
        return f"Edge(start={self.start}, stop={self.stop})"

    def __repr__(self) -> str:
        return str(self)


class Label:
    def __init__(self, topLeft, dimensions, content) -> None:
        self.topLeft = topLeft
        self.width = dimensions[0]
        self.height = dimensions[1]
        self.content = content

    def __str__(self) -> str:
        return f"Label(topLeft={self.topLeft}, dimensions=({self.width}, {self.height}), content={self.content})"

    def __repr__(self) -> str:
        return str(self)
