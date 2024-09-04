# SPDX-FileCopyrightText: 2024 Julius Deynet <jdeynet@googlemail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from detect_graph import harvest_graph
import json
import unittest
import math


class TestGraphHarvester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGraphHarvester, self).__init__(*args, **kwargs)
        with open("backend/tests/test_data/input.json") as f:
            self.data = json.load(f)

        self.result = harvest_graph(self.data, False, True)

        with open("backend/tests/test_data/expected_result.json") as f:
            self.expected = json.load(f)

    def compare_json(self, json1, json2, float_tolerance=1e-9):
        if isinstance(json1, dict) and isinstance(json2, dict):
            self.assertEqual(json1.keys(), json2.keys())

            return all(
                self.compare_json(json1[key], json2[key], float_tolerance)
                for key in json1
            )

        elif isinstance(json1, list) and isinstance(json2, list):
            self.assertEqual(len(json1), len(json2))

            return all(
                self.compare_json(item1, item2, float_tolerance)
                for item1, item2 in zip(json1, json2)
            )

        elif isinstance(json1, float) and isinstance(json2, float):
            if json1 - json2 > float_tolerance:
                pass
            return self.assertAlmostEqual(json1, json2, delta=float_tolerance)

        else:
            return self.assertEqual(json1, json2)

    def test_harvest_graph(self):
        for key, value in self.result[0].items():
            expected_value = self.expected[0][key]
            self.compare_json(value, expected_value)

    def test_harvest_graph_with_curves(self):
        for key, value in self.result[1].items():
            expected_value = self.expected[1][key]
            self.compare_json(value, expected_value)
