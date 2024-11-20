# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import logging
import re
from copy import copy
from typing import Dict, Iterable, List, Optional, Tuple

from kieta_data_objs import BoundingBox, Document

from . import util
from .base import Module

logger = logging.getLogger('main')


def roman_to_arabic(roman_numeral):
    roman_numerals = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    prev_value = 0
    
    for numeral in reversed(roman_numeral):
        value = roman_numerals[numeral]
        
        if value < prev_value:
            total -= value
        else:
            total += value
        
        prev_value = value
    
    return total


class KeywordCaptionDetector(Module):
    _MODULE_TYPE = 'KeywordCaptionDetector'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        if isinstance(parameters["keywords"], list):
            self.keywords = parameters["keywords"]
        else:
            self.keywords = [x.strip() for x in parameters.get("keywords", "").split(',')]
        if isinstance(parameters["delimiters"], list):
            self.delimiters = parameters["delimiters"]
        else:
            self.delimiters = [x.strip() for x in parameters.get("delimiters", "").split(',')]
        # self.distance_line_above = int(parameters["distance_line_above"])
        self.caption_expansion_threshold = int(parameters.get("expansion_threshold", 5))
        self.removal = parameters.get("removal", [])
        self.keep_most_common = bool(parameters.get("keep_most_common", False))

        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

    
    def get_text_lines(self, doc: Document) -> Iterable[Tuple[str, List[str]]]:
        for l in doc.get_area_type(self.lines):
            # aggregate line
            line_string = list()
            line_objs = list()
            try:
                for xx in doc.references.byId[l.oid]:
                    try:
                        line_string.append(' '.join(doc.get_area_data_value(xx, 'content')))
                        line_objs.append(xx)
                    except KeyError:
                        pass
            except KeyError:
                pass
            yield ' '.join(line_string), line_objs

    def execute(self, inpt: Document) -> Document:
        """
        Detects table captions within document using pattern matching. Assumes that delimiter between identifier and caption string is the same for all tables
        """
        adds = list()

        for s, lines in self.get_text_lines(inpt):
            if (t := self.check_pattern(s))[0]:
                # print(f"Found caption {s}")
                # search for next line below with threshold
                #get page
                page_id = inpt.find_page_of_area(lines[0])

                # todo: rotated tables are not working
                horizontal_lines, vertical_lines = util.sort_into_two_lists(inpt.get_area_type('DrawnLine', page_id), lambda x: x.boundingBox.is_horizontal())

                current = None
                content = s
                refs = list()
                for x in lines:
                    if current is None:
                        current = BoundingBox(inpt.areas.byId[x].boundingBox.x1, 
                                                inpt.areas.byId[x].boundingBox.y1,
                                                inpt.areas.byId[x].boundingBox.x2,
                                                inpt.areas.byId[x].boundingBox.y2,
                                                inpt.areas.byId[x].boundingBox.img_sp)
                    else:
                        current.expand(inpt.areas.byId[x].boundingBox)
                    try:
                        refs.extend(inpt.references.byId[x])
                    except KeyError:
                        pass
                    refs.append(x)

                stats = list()
                for other in list(inpt.get_area_type(self.lines, page_id)) + list(inpt.get_area_type(self.strings, page_id)):
                    # find below objects
                    if current.img_sp:
                        other_bb = other.boundingBox.get_in_img_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                    else:
                        other_bb = other.boundingBox.get_in_xml_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                    # print(other, other_bb, current)

                    other_content = ''.join(inpt.get_area_data_value(other, 'content'))

                    stats.append(
                    (other_content, content, other_bb.y1 - current.y2, other_bb.overlap_horizontally(current))
                    )
                    
                    if (abs(current.y1 - other_bb.y2) < self.caption_expansion_threshold or \
                    abs(other_bb.y1 - current.y2) < self.caption_expansion_threshold ) and \
                            other_bb.overlap_horizontally(current):

                        # ensure that there is no drawn line between current and other
                        # if there is, do not expand
                        # if not, expand
                        mock_expand = copy(current)
                        mock_expand.expand(other_bb)
                        found_intersect = False
                        for l in horizontal_lines:
                            if current.img_sp:
                                bounding = l.boundingBox.get_in_img_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                            else:
                                bounding = l.boundingBox.get_in_xml_space(inpt.pages.byId[page_id].factor_width, inpt.pages.byId[page_id].factor_height)
                            if bounding.intersects(mock_expand):
                                found_intersect = True
                        if found_intersect:
                            continue

                        current.expand(other_bb)
                        try:
                            refs.extend(inpt.references.byId[other])
                        except KeyError:
                            pass
                        refs.append(other)
                        try:
                            content += inpt.areas.byId[other].data['content']
                        except KeyError:
                            pass
                
                # for x in list(sorted(stats, key=lambda x: x[2])):
                #     print(x)
                # if current == lines:
                #     print(f"couldn't expand {s}")
                adds.append((page_id, current, content, t[1], refs))

                # check that distance to next line above is above threshold
                # other_lines = list()
                # for other in inpt.references.byId[f'page-{page}']:
                #     if inpt.areas.byId[other].boundingBox.overlap_vertically(current) and \
                #             current.y1 > inpt.areas.byId[other].boundingBox.y2:
                #         other_lines.append((inpt.areas.byId[other].boundingBox, current.y1 - inpt.areas.byId[other].boundingBox.y2 ))
                #
                # if min(other_lines, key=lambda x: x[1])[1] > self.distance_line_above:
                #     adds.append((f'page-{page}', current, content))


        # count combinations of (delimiter, keyword)
        # keep only most common
        captions_stats = collections.Counter([a[3] for a in adds])
        logger.debug(f"{inpt.oid} : Found {captions_stats} captions")
        try:
            most_common = captions_stats.most_common(1)[0][0]
        except IndexError:
            most_common = -1
        ix = 0

        for a in adds:
            if self.keep_most_common and a[3] != most_common:
                continue
            matches = re.findall(r"\b(?:\d+|[IVX]+)\b", a[2])
            
            ix += 1

            if len(matches) > 0:
                table_no = matches[0]
                # convert roman numerals to arabic
                if table_no.isupper():
                    self.debug_msg(f"Found roman numeral {table_no} converting to arabic")
                    table_no = roman_to_arabic(table_no)
            else:
                table_no = ix
                logger.info('did not find table number, using table caption index')
            self.debug_msg(f'TABLE NUMBER {table_no}')


            if any([x.lower() in a[2].lower() for x in self.removal]):
                # self.debug_msg(f"Found continued table {a[2]}")
                inpt.add_area(page=a[0], category='Caption', boundingBox=a[1], data={'content': a[2], 'number': table_no, 'continued': True}, references=a[4])
            else:
                inpt.add_area(page=a[0], category='Caption', boundingBox=a[1], data={'content': a[2], 'number': table_no}, references=a[4])
        # self.debug_msg(f"Found {ix} captions")
        # TODO: captions cannot cross drawn lines

        return inpt

    def check_pattern(self, s: str) -> Tuple[bool, Tuple[str, int]]:
        """
        Checks, if string fits pattern in patternstorage, returns delimiter

        :param s: string to check
        :return: [Result of check, delimiter]
        """

        sp = s.split(' ')
        if len(sp) >= 4:
            sp = sp[:4]

        try:
            sp.remove('')
        except ValueError:
            pass

        sp = [x for x in sp if x not in self.removal and x != '']

        if len(sp) == 0:
            return False, ("", -1)

        # make every substring combination of sp[0]
        combinations = set([sp[0]] + [sp[0][0:x] for x in range(1, len(sp[0]))])
        kw_index = -1
        if not sp:
            return False, ("", -1)
        elif sp[0].isupper():  # todo: no idea if that's working
            try:
                return True, ("", self.keywords.index(sp[0]))
            except ValueError:
                pass
        
        if (t:=set(self.keywords).intersection(combinations)):
            # index of keyword
            try:
                kw_index = self.keywords.index(list(t)[0])
            except Exception:
                pass
        else:
            return False, ("", -1)

        for x in range(len(sp)):
            for r in self.removal:
                sp[x] = sp[x].replace(r, '')

        try:
            if not (sp[-1][0].isupper() or sp[-2][0].isupper()):
                return False, ("", -1)
        except IndexError:
            pass
        
        try:
            for x in sp[1:]:
                if x[-1].isdigit():
                    return True, ('\n',kw_index)

                for p in self.delimiters:
                    if x[-1] == p:
                        return True, (p, kw_index)
        except IndexError as e:
            logger.error(f"Error in check_table_pattern: {e}")
            quit()

        # make the same for only sp[0]
        if sp[0][-1] in self.delimiters:
            return True, (sp[0][-1], kw_index)

        return False, ("", -1)


class WhitespaceCaptionDetector(Module):
    _MODULE_TYPE = 'WhitespaceCaptionDetector'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.direction = parameters.get('direction', 'above')  # below, TODO: auto

        self.distance_to_table_threshold = int(parameters.get('distance_to_table_threshold', 5))  # how much distance between caption and table
        self.distance_to_next_object = int(parameters.get('distance_to_next_object', 5)) # how much distance between caption and next object

        self.apply_to = parameters.get('apply_to', 'Line')

        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')
    
    def execute(self, inpt: Document) -> Document:
        to_be_added = list()
        for tab in inpt.get_area_type('Table'):
            # get page
            page_id = inpt.find_page_of_area(tab.oid)
            # get table bounding box
            table_bb = inpt.areas[tab.oid].boundingBox.get_in_img_space(inpt.pages[page_id].factor_width, inpt.pages[page_id].factor_height)
            # get all lines
            lines = list()
            def calculate_distance(x, y):
                return x.boundingBox.y1 - y.boundingBox.y2
            x_range = range(int(table_bb.x1), int(table_bb.x2))

            for element in inpt.get_area_type(self.apply_to, page_id):
                if element.boundingBox.x1 in x_range or element.boundingBox.x2 in x_range:
                    lines.append(element)

            match self.direction:
                case 'above':
                    # get all lines above table
                    lines = list(sorted([x for x in lines if x.boundingBox.y2 < table_bb.y1], key=lambda x: x.boundingBox.y2, reverse=True))
                case 'below': 
                    # get all lines below table
                    lines = list(sorted([x for x in lines if x.boundingBox.y1 > table_bb.y2]), key=lambda x: x.boundingBox.y1)
                case _:
                    raise ValueError(f"Invalid direction {self.direction}")
                
            # check if the distance from first line to table is above threshold
            if len(lines) == 0:
                self.debug_msg(f"No line found above/below table {tab.oid}")
                continue
            first_line = lines[0]

            if calculate_distance(first_line, tab) >= self.distance_to_table_threshold:
                self.debug_msg(f"Distance from line to table above threshold {first_line.oid} {tab.oid}")
                continue
            
            caption = [first_line]
            caption_bb = BoundingBox(caption[0].boundingBox.x1, caption[0].boundingBox.y1, caption[-1].boundingBox.x2, caption[-1].boundingBox.y2, caption[0].boundingBox.img_sp)

            # expand as long as distance is below "distance_to_next_object"
            for line in lines[1:]:
                if calculate_distance(line, caption[-1]) < self.distance_to_next_object:
                    caption.append(line)
                    caption_bb.expand(line.boundingBox)
                else:
                    break

            caption = util.sort_2D_grid(caption)[0]

            # get content of caption
            content = []
            for c in caption:
                try:
                    content += inpt.get_area_data_value(c.oid, 'content')
                except KeyError:
                    pass
            content = ' '.join(content)
            
            # add caption
            to_be_added.append(
                ({"page":page_id, "category":'Caption', "boundingBox":caption_bb, "data":{'content': content}, "references":[x.oid for x in caption], "convert_to_xml":False}, tab))

        for add, tab in to_be_added:
            tt = inpt.add_area(**add)
            inpt.areas.byId[tab.oid].data['caption'] = tt

        return inpt