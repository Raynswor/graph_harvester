# SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from geometry_util import *


class TestUtil(unittest.TestCase):
    def test_detect_circles_from_beziers(self):
        beziers = [
            Bezier(
                (1.04753112792969, 0.00880126953125),
                (0.04753112792969, 0.95135498046875),
                (0.02791595458984, 0.93174743652344),
                (2.07047729492188, 1.03174743652344),
            ),
            Bezier(
                (2.07047729492188, 1.03174743652344),
                (0.5130386352539, 0.93174743652344),
                (0.49342346191406, 0.95135498046875),
                (1.09342346191406, 2.00880126953125),
            ),
            Bezier(
                (1.09342346191406, 2.00880126953127),
                (0.49342346191406, 0.46624755859375),
                (0.5130386352539, 0.48585510253906),
                (0.07047729492188, 1.08585510253906),
            ),
            Bezier(
                (0.07047729492188, 1.08585510253906),
                (0.02791595458984, 0.48585510253906),
                (0.04753112792969, 0.46624755859375),
                (1.04753112792969, 0.00880126953125),
            ),
        ]

        circles, beziers = detect_circles_from_beziers(beziers)

        self.assertTrue(len(circles) == 1)
        self.assertTrue(np.allclose(circles[0].center, (1, 1), atol=1e-1))
        self.assertTrue(np.isclose(circles[0].radius, 1, atol=1e-1))

        self.assertTrue(len(beziers) == 0)

        oval_beziers = [
            Bezier(
                (1.04753112792969, 0.00880126953125),
                (0.04753112792969, 0.95135498046875),
                (0.02791595458984, 0.93174743652344),
                (3.07047729492188, 1.03174743652344),
            ),
            Bezier(
                (3.07047729492188, 1.03174743652344),
                (0.5130386352539, 0.93174743652344),
                (0.49342346191406, 0.95135498046875),
                (1.09342346191406, 2.00880126953125),
            ),
            Bezier(
                (1.09342346191406, 2.00880126953127),
                (0.49342346191406, 0.46624755859375),
                (0.5130386352539, 0.48585510253906),
                (-1.07047729492188, 1.08585510253906),
            ),
            Bezier(
                (-1.07047729492188, 1.08585510253906),
                (0.02791595458984, 0.48585510253906),
                (0.04753112792969, 0.46624755859375),
                (1.04753112792969, 0.00880126953125),
            ),
        ]

        circles, beziers = detect_circles_from_beziers(oval_beziers)

        self.assertTrue(len(circles) == 0)
        self.assertTrue(len(beziers) == 4)

        three_beziers = [
            Bezier(
                (884.0607581899948, 891.9788648917952),
                (884.050200646665, 881.5107455038401),
                (872.7772247570904, 874.9761355801036),
                (863.7455644367122, 880.2041541890123),
            ),
            Bezier(
                (863.7455644367122, 880.2041541890123),
                (854.7161301043856, 885.466702368367),
                (854.73152122177, 898.5373212039611),
                (863.761400751707, 903.7671839335751),
            ),
            Bezier(
                (863.761400751707, 903.7671839335751),
                (872.7996118369223, 908.9828660108698),
                (884.0519178374475, 902.4499730271001),
                (884.0607581899948, 891.9819808198833),
            ),
        ]

        circles, beziers = detect_circles_from_beziers(three_beziers)

        self.assertTrue(len(circles) == 1)
        self.assertTrue(np.allclose(circles[0].center, (870.5, 891.9), atol=1e-1))
        self.assertTrue(np.isclose(circles[0].radius, 13.5, atol=1e-1))

        self.assertTrue(len(beziers) == 0)

    def test_get_circle_containing_point(self):
        circles = [Circle(1, (1, 1))]

        self.assertEqual(get_circle_containing_point(circles, (1, 1)), circles[0])

        self.assertEqual(get_circle_containing_point(circles, (2, 2)), circles[0])

        self.assertEqual(get_circle_containing_point(circles, (3, 3)), None)

    def test_circles_on_line(self):
        circles = [
            Circle(0.1, (1, 1)),
            Circle(0.1, (3, 3)),
            Circle(0.1, (2, 2)),
            Circle(0.1, (4, 4)),
        ]
        line = Line((1, 1), (3, 3))
        self.assertEqual(
            circles_on_line(circles, line), [circles[0], circles[2], circles[1]]
        )

        line = Line((4, 0), (6, 0))
        self.assertEqual(circles_on_line(circles, line), [])

        # Circles right next to line should not be detected, only the ones that intersect
        circles = [Circle(0.1, (1, 1)), Circle(0.1, (1, 3)), Circle(0.1, (1.11, 2))]
        line = Line((1, 1), (1, 3))
        self.assertEqual(circles_on_line(circles, line), [circles[0], circles[1]])

    def test_circles_on_bezier(self):
        circles = [
            Circle(0.1, (1, 1)),
            Circle(0.1, (3, 3)),
            Circle(0.1, (2, 2)),
            Circle(0.1, (4, 4)),
        ]
        bezier = Bezier((1, 1), (2, 2), (3, 3), (4, 4))
        self.assertEqual(
            circles_on_bezier(circles, bezier), [circles[0], circles[2], circles[1]]
        )

        bezier = Bezier((4, 0), (4.5, 0), (5.5, 0), (6, 0))
        self.assertEqual(circles_on_bezier(circles, bezier), [])

    def test_detect_edges_from_lines(self):
        circles = [
            Circle(0.1, (10, 10)),
            Circle(0.1, (30, 30)),
            Circle(0.1, (20, 20)),
            Circle(0.1, (40, 40)),
        ]
        lines = [Line((10, 10), (30, 30)), Line((40, 0), (60, 0))]
        result = detect_edges_from_lines(circles, lines, [])
        self.assertEqual(
            result,
            [
                Edge(circles[0], circles[2]),
                Edge(circles[2], circles[0]),
                Edge(circles[2], circles[1]),
                Edge(circles[1], circles[2]),
            ],
        )
        self.assertTrue(lines[0].used)
        self.assertFalse(lines[1].used)

        # Extending lines works
        lines = [Line((10, 10), (10, 20)), Line((10, 20), (20, 20))]
        result = detect_edges_from_lines(circles, lines, [])
        self.assertEqual(
            result,
            [Edge(circles[0], circles[2]), Edge(circles[2], circles[0])],
        )
        self.assertTrue(lines[0].used)
        self.assertTrue(lines[1].used)

    def test_detect_edges_from_beziers(self):
        circles = [
            Circle(0.1, (10, 10)),
            Circle(0.1, (30, 30)),
            Circle(0.1, (20, 20)),
            Circle(0.1, (40, 40)),
        ]
        beziers = [
            Bezier((10, 10), (25, 25), (35, 35), (30, 30)),
            Bezier((40, 0), (45, 0), (55, 0), (60, 0)),
        ]
        lines = []

        self.assertEqual(
            detect_edges_from_beziers(circles, beziers, lines, []),
            [
                Edge(circles[0], circles[2]),
                Edge(circles[2], circles[0]),
                Edge(circles[2], circles[1]),
                Edge(circles[1], circles[2]),
            ],
        )

        # Extending beziers by another bezier works
        beziers = [
            Bezier((10, 10), (10, 13), (10, 17), (10, 20)),
            Bezier((10, 20), (13, 20), (17, 20), (20, 20)),
        ]
        self.assertEqual(
            detect_edges_from_beziers(circles, beziers, lines, []),
            [Edge(circles[0], circles[2]), Edge(circles[2], circles[0])],
        )

        # Extending beziers by line works
        beziers = [
            Bezier((10, 10), (10, 7), (10, 3), (10, 0)),
            Bezier((10, 0), (13, 0), (17, 0), (20, 0)),
        ]
        lines = [Line((20, 0), (20, 20))]
        self.assertEqual(
            detect_edges_from_beziers(circles, beziers, lines, []),
            [Edge(circles[0], circles[2]), Edge(circles[2], circles[0])],
        )

        # bezier -> line -> bezier -> line with mixed order works
        beziers = [
            Bezier((10, 0), (13, 0), (17, 0), (20, 0)),
            Bezier((10, 10), (7, 7), (3, 3), (0, 0)),
        ]
        lines = [Line((20, 0), (20, 20)), Line((0, 0), (10, 0))]
        self.assertEqual(
            detect_edges_from_beziers(circles, beziers, lines, []),
            [Edge(circles[0], circles[2]), Edge(circles[2], circles[0])],
        )

    def test_filter_duplicate_circles(self):
        # Overlapping
        circles = [
            Circle(1, (1, 1)),
            Circle(1, (1, 1.5)),
            Circle(1, (3, 3)),
        ]
        filtered_circles = filter_duplicate_circles(circles)
        self.assertEqual(filtered_circles, [circles[0], circles[2]])

        # Tripple overlapping
        circles = [
            Circle(1, (1, 1)),
            Circle(1, (1.25, 1.25)),
            Circle(1, (0.75, 0.75)),
        ]
        filtered_circles = filter_duplicate_circles(circles)
        self.assertEqual(filtered_circles, [circles[0]])

        # One circle containg two others
        circles = [
            Circle(1, (1, 1)),
            Circle(1, (1.25, 1.25)),
            Circle(5, (3, 3)),
        ]
        filtered_circles = filter_duplicate_circles(circles)
        self.assertEqual(filtered_circles, [circles[2]])

        # Two equal circles
        circles = [Circle(1, (1, 1)), Circle(1, (1, 1))]
        filtered_circles = filter_duplicate_circles(circles)
        self.assertEqual(filtered_circles, [circles[0]])

    def test_filter_circles_based_on_size(self):
        dummy_bezier = Bezier((1, 1), (2, 2), (3, 3), (4, 4))
        circles = [
            Circle(1, (1, 1)),
            Circle(1, (3, 3)),
            Circle(1, (5, 5)),
            Circle(0.5, (1, 1), True, [dummy_bezier]),
        ]
        filtered_circles, regained_beziers = filter_circles_based_on_peak_size(
            circles, [], []
        )
        self.assertEqual(filtered_circles, circles[:3])

        self.assertEqual(regained_beziers, [dummy_bezier])

    def test_rect_on_vertices(self):
        rects = [Rect((1, 1), (2.1, 2.1), False)]
        circles = [
            Circle(0.1, (1, 1)),
            Circle(0.1, (2, 2)),
            Circle(0.1, (1, 2)),
            Circle(0.1, (2, 1)),
        ]
        new_rects, lines = try_converting_rect_to_lines(rects, circles)
        self.assertEqual(new_rects, [])
        self.assertEqual(
            lines,
            [
                Line((1, 1), (1, 2.1)),
                Line((1, 1), (2.1, 1)),
                Line((1, 2.1), (2.1, 2.1)),
                Line((2.1, 1), (2.1, 2.1)),
            ],
        )

        rects = [Rect((1, 1), (3, 3), False)]
        new_rects, lines = try_converting_rect_to_lines(rects, circles)
        self.assertEqual(new_rects, rects)
        self.assertEqual(
            lines,
            [],
        )

        # ignore filled rects
        rects = [Rect((1, 1), (2.1, 2.1), True)]
        new_rects, lines = try_converting_rect_to_lines(rects, circles)
        self.assertEqual(new_rects, rects)
        self.assertEqual(
            lines,
            [],
        )

        rects = [Rect((1, 1), (3, 3), False)]
        circles = circles[:2]
        new_rects, lines = try_converting_rect_to_lines(rects, circles)
        self.assertEqual(new_rects, rects)
        self.assertEqual(
            lines,
            [],
        )

    def test_compute_subgraphs(self):
        adjacency_matrix = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 1, 0],
            ]
        )
        subgraphs = compute_subgraphs(adjacency_matrix)

        self.assertEqual(len(subgraphs), 2)

        self.assertTrue(np.array_equal(subgraphs[0], np.array([[0, 1], [1, 0]])))
        self.assertTrue(
            np.array_equal(subgraphs[1], np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        )

        adjacency_matrix = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
            ]
        )
        subgraphs = compute_subgraphs(adjacency_matrix)

        self.assertEqual(len(subgraphs), 1)

        self.assertTrue(np.array_equal(subgraphs[0], adjacency_matrix))

    def test_max_node_degree(self):
        adjacency_matrix = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
            ]
        )

        self.assertEqual(max_node_degree(adjacency_matrix), 2)

        adjacency_matrix = np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [1, 1, 0, 1, 0],
            ]
        )

        self.assertEqual(max_node_degree(adjacency_matrix), 3)

        adjacency_matrix = np.ones((5, 5))
        self.assertEqual(max_node_degree(adjacency_matrix), 4)
