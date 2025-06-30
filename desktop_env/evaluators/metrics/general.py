import csv
import datetime
import difflib
import functools
import json
import logging
import operator
import os
import re
import sqlite3
from numbers import Number
from typing import Callable, Any, Union
from typing import Dict, List, Pattern

import lxml.etree
import pdfplumber
import yaml
from docx import Document
from lxml.cssselect import CSSSelector
from lxml.etree import _Element
from rapidfuzz import fuzz
import filecmp
import hashlib
import subprocess
from .chrome import compare_pdfs

from desktop_env.evaluators.metrics.utils import _match_record, _match_value_to_rule

logger = logging.getLogger("desktopenv.metric.general")


def check_include_exclude(result: str, rules: Dict[str, List[str]]) -> float:
    if result is None:
        return 0.

    print(result, rules)
    include = rules.get("include", [])
    exclude = rules.get("exclude", [])
    if all(r in result for r in include) and all(r not in result for r in exclude):
        return 1.
    else:
        return 0.


def check_one_of_include(result: list, rules: Dict[str, List[str]]):
    if result is None:
        return 0.
    include = rules.get("expected", [])
    if isinstance(include, str):
        include = [include]
    for item in result:
        if all(r in item for r in include):
            return 1.
    else:
        return 0.
    

def check_one_of_equal(result: list, rules: Dict[str, List[str]]):
    if result is None:
        return 0.
    expected = rules.get("expected", [])
    return float(any(item == expected for item in result))


def exact_match(result, rules) -> float:
    expect = rules["expected"]
    print(result, expect)

    if result == expect:
        return 1.
    else:
        return 0.


def literal_match(result: Any, expected: Any, **options) -> float:
    literal_type = options.get('type', 'str')
    if literal_type == 'str':
        ignore_case = options.get('ignore_case', False)
        score = str(result) == str(expected) if not ignore_case else str(result).lower() == str(expected).lower()
        return float(score)
    elif literal_type == 'list':
        if type(result) not in [list, tuple] or type(expected) not in [list, tuple] or len(result) != len(expected):
            return .0
        ignore_case = options.get('ignore_case', False)
        result = [str(s) for s in result] if not ignore_case else [str(s).lower() for s in result]
        expected = [str(s) for s in expected] if not ignore_case else [str(s).lower() for s in expected]
        return float(result == expected)
    else:
        raise NotImplementedError(f"Type {type} not supported")


def is_in_list(result, rules) -> float:
    expect = rules["expected"]
    if expect in result:
        return 1.
    else:
        return 0.


def diff_text_file(result: str, expect: str) -> float:
    if result is None:
        return 0.

    with open(result) as f:
        result_lines: List[str] = f.read().splitlines()
    with open(expect) as f:
        expected_lines: List[str] = f.read().splitlines()
    return difflib.SequenceMatcher(a=result_lines, b=expected_lines).ratio()


def fuzzy_match(result, rules) -> float:
    expect = rules["expected"]

    return fuzz.ratio(result, expect) / 100.


def fuzzy_place_math(result_file_path, rules) -> float:
    if result_file_path is None:
        return 0.
    expect = rules["expected"]  # a list of possible answers
    # read list.docx, and get all texts out, overlook blank lines, remove blanks before and after each line
    doc = Document(result_file_path)
    words_list = []
    for para in doc.paragraphs:
        words_list.extend(para.text.split())
    fuzzy_score_list = []
    for word in words_list:
        max_score = 0
        for ans in expect:
            score = fuzz.ratio(word, ans) / 100
            max_score = max(max_score, score)
        fuzzy_score_list.append(max_score)
    if len(fuzzy_score_list) != 3:
        return 0.
    return sum(fuzzy_score_list) / 3


def check_csv(result: str, rules: Dict[str, List[Dict[str, str]]]) -> float:
    """
    Args:
        result (str): path to csv file
        rules (Dict[str, List[Dict[str, str]]]): dict like
          {
            "expect": [{key: value}]
            "unexpect": [{key: value}]
          }

    Returns:
        float
    """

    if result is None:
        return 0.

    expect_metrics = [False] * len(rules.get("expect", []))
    unexpect_metric = True
    with open(result) as f:
        reader = csv.DictReader(f)

        for rcd in reader:
            for i, r in enumerate(rules.get("expect", [])):
                expect_metrics[i] = expect_metrics[i] or _match_record(r, rcd)
            unexpect_metric = unexpect_metric and not any(_match_record(r, rcd) for r in rules.get("unexpect", []))
    return float(all(expect_metrics) and unexpect_metric)


def check_list(result: str, rules: Dict[str, List[str]]) -> float:
    """
    Args:
        result (str): path to list file
        rules (Dict[str, List[str]]): dict like
          {
            "expect": list of str as regexes
            "unexpect": list of str as regexes
          }

    Returns:
        float
    """

    if result is None:
        return 0.

    expect_patterns: List[Pattern[str]] = [re.compile(ptt) for ptt in rules.get("expect", [])]
    unexpect_patterns: List[Pattern[str]] = [re.compile(ptt) for ptt in rules.get("unexpect", [])]

    expect_metrics = [False] * len(expect_patterns)
    unexpect_metric = True
    with open(result) as f:
        for l in f:
            for i, r in enumerate(expect_patterns):
                expect_metrics[i] = expect_metrics[i] or (r.search(l) is not None)
            unexpect_metric = unexpect_metric and all(r.search(l) is None for r in unexpect_patterns)
    return float(all(expect_metrics) and unexpect_metric)


_accessibility_ns_map = {"st": "uri:deskat:state.at-spi.gnome.org"
    , "attr": "uri:deskat:attributes.at-spi.gnome.org"
    , "cp": "uri:deskat:component.at-spi.gnome.org"
    , "doc": "uri:deskat:document.at-spi.gnome.org"
    , "docattr": "uri:deskat:attributes.document.at-spi.gnome.org"
    , "txt": "uri:deskat:text.at-spi.gnome.org"
    , "val": "uri:deskat:value.at-spi.gnome.org"
    , "act": "uri:deskat:action.at-spi.gnome.org"
                         }


def check_accessibility_tree(result: str, rules: List[Dict[str, Any]]) -> float:
    """
    Args:
        result (str): XML of GNOME Accessibility Tree
        rules (List[Dict[str, Any]]): list of dict like
          {
            "selectors": list of str as CSS selectors, will be connected by ", "
              to form a composite selector. Only one from `selectors` and
              `xpath` is needed. If both are present, `xpath` takes the
              priority.
            "xpath": str as xpath. Only one from `selectors` and `xpath` is
              needed. If both are present, `xpath` takes the priority.
            "text": str as the expected text content of the selected element.
            "exact": bool specifying whether exact match or fuzzy match should
              be performed. defaults to True.
          }

    Returns:
        float
    """

    at: _Element = lxml.etree.fromstring(result)
    total_match_score = 1.
    for r in rules:
        if "xpath" in r:
            elements: List[_Element] = at.xpath(r["xpath"], namespaces=_accessibility_ns_map)
        elif "selectors" in r:
            selector = CSSSelector(", ".join(r["selectors"]), namespaces=_accessibility_ns_map)
            elements: List[_Element] = selector(at)
        else:
            raise ValueError("At least one of xpath and selectors is required")

        if len(elements) == 0:
            logger.info("No elements: %s", r["xpath"] if "xpath" in r else r["selectors"])
            return 0.

        if "text" in r:
            match_func: Callable[[str], Number] = functools.partial(operator.eq if r["exact"] \
                                                                        else (lambda a, b: fuzz.ratio(a, b) / 100.)
                                                                    , r["text"]
                                                                    )
            match_score: Number = 0
            for elm in elements:
                match_score = max(match_score, match_func(elm.text or None))
        else:
            match_score = 1.
        total_match_score *= match_score

    return float(total_match_score)


# def check_existence(result: str, *args) -> float:
# return 1. - (result is None)

def run_sqlite3(result: str, rules: Dict[str, Any]) -> float:
    connection: sqlite3.Connection = sqlite3.connect(result)
    cursor: sqlite3.Cursor = connection.execute(rules["sql"])
    return float(cursor.fetchone()[0] or 0)


def check_json(result: str, rules: Dict[str, List[Dict[str, Union[List[str], str]]]], is_yaml: bool = False) -> float:
    """
    Args:
        result (str): path to json file
        rules (Dict[str, List[Dict[str, Union[List[str], str]]]]): dict like
          {
            "expect": [
                {
                    "key": list of str
                    "method": str
                    "ref": something
                }
            ],
            "unexpect": <the same as `expect`
          }
        is_yaml (bool): yaml rather than json

    Returns:
        float
    """

    if result is None:
        return 0.
    with open(result) as f:
        if is_yaml:
            result: Dict[str, Any] = yaml.load(f, Loader=yaml.Loader)
        else:
            result: Dict[str, Any] = json.load(f)

    expect_rules = rules.get("expect", {})
    unexpect_rules = rules.get("unexpect", {})

    metric = True
    for r in expect_rules:
        value = result
        for k in r["key"]:
            try:
                value = value[k]
            except KeyError:
                return 0.
        metric = metric and _match_value_to_rule(value, r)
    for r in unexpect_rules:
        value = result
        for k in r["key"]:
            try:
                value = value[k]
            except KeyError:
                value = None
                break
        metric = metric and not _match_value_to_rule(value, r)
    return float(metric)


def check_json_contains(result, rules):
    expected = rules["expected"]
    for key, value in expected.items():
        if value not in result.get(key, []):
            return 0.0
    return 1.0


def check_direct_json_object(result, rules) -> float:
    """
    One of the most commonly used function to evalute.
    Compare two json objects directly.
    """
    if isinstance(result, str):
        # remove blanks before and after result
        result = result.strip()
        # replace all ' with "
        result = result.replace("'", '\'')
        # load json object
        result = json.loads(result)
    if result is None:
        return 0.
    try:
        expect_in_result = rules.get("expect_in_result", False)
        if not expect_in_result:
            expected_json = rules["expected"]
            for key in expected_json.keys():
                expected_value = expected_json.get(key)
                if expected_value != result.get(key):
                    return 0.
            return 1.0
        else:
            expected_json = rules["expected"]

            for key in expected_json.keys():
                if isinstance(expected_json.get(key), list):
                    flag = 0
                    expected_value_list = expected_json.get(key)
                    for each_expected_value in expected_value_list:
                        if isinstance(result.get(key), list) and each_expected_value in result.get(key):
                            flag = 1
                            break
                    if flag == 0:
                        return 0.
                elif isinstance(expected_json.get(key), str):
                    if expected_json.get(key) not in result.get(key):
                        return 0.
                else:
                    logger.debug("check_direct_json_object: expected value type not supported")
                    return 0.
            return 1.0
    except:
        logger.debug("check_direct_json_object: result is not a valid json object")
        return 0.


def compare_time_in_speedtest_results(speedtest_result_path, time_diff):
    if not speedtest_result_path:
        return 0

    # open the speedtest results file(csv)
    date_col = None
    try:
        with open(speedtest_result_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 1:
                    date = line.split(',')[1]
                    break
            now_date_time = datetime.datetime.now().strftime('%H:%M')
            date_time = date[-5:]
            # compare the date time with the current date time, if time diff less than time_diff para, then return true
            if not abs((datetime.datetime.strptime(date_time, '%H:%M') - datetime.datetime.strptime(now_date_time,
                                                                                                    '%H:%M')).total_seconds()) / 60 < int(
                time_diff):
                return 0
        return 1
    except:
        logger.debug("compare_time_in_speedtest_results: file not found or not readable")
        return 0


def is_included_all_json_objects(gold_file_path, result_file_path):
    if not gold_file_path or not result_file_path:
        return 0

    print("gold_file_path: ")
    print(gold_file_path)
    print("result_file_path: ")
    print(result_file_path)
    # two json file, check if all the key-value pair in gold_file_path is included in result_file_path
    with open(gold_file_path, 'r') as f:
        gold_json = json.load(f)
    with open(result_file_path, 'r') as fr:
        result_json = json.load(fr)
    for key in gold_json.keys():
        if key not in result_json.keys() or gold_json[key] != result_json[key]:
            return 0
    return 1


def is_gold_text_included_in_pdf(pdf_file_path, gold_text_path):
    if not gold_text_path or not pdf_file_path:
        return 0

    print("gold_text_path: ")
    print(gold_text_path)
    print("pdf_file_path: ")
    print(pdf_file_path)
    # gold file is a json file, we need to check all the value in json are included in pdf file.
    with open(gold_text_path, 'r') as f:
        gold_json = json.load(f)
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    false_list = []
    for key in gold_json.keys():
        if gold_json[key] not in text:
            false_list.append(key)
    if len(false_list) > 0:
        print("false_list: ")
        print(false_list)
        return 0
    else:
        return 1


def file_contains(file_path, config):
    # file_path ends with .txt
    if not file_path:
        return 0.
    try:
        with open(file_path, 'r') as f:
            file_text = f.read()
        if isinstance(config["expected"], str):
            config["expected"] = [config["expected"]]
        for text in config["expected"]:
            if text not in file_text:
                logger.debug(f"file_contains: {text} not found in {file_path}")
                return 0.
    except:
        logger.debug("file_contains: file not found or not readable")
        return 0.
    return 1.


def check_line_number(file_path, line_number):
    # check if file_path exists
    if file_path is None or not os.path.isfile(file_path):
        return 0.
    timeRegex = "([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)"
    # check if the string that matches the timeRegex in this txt file equals to line_number["expected"]
    try:
        with open(file_path, 'r') as f:
            line_count = 0
            for line in f:
                if re.search(timeRegex, line):
                    line_count += 1
        # if line_count equals to line_number["expected"], return 1, else return 0
        return 1 if line_count == int(line_number["expected"]) else 0
    except:
        logger.debug("check_line_number: file not found or not readable")
        return 0.


def compare_terminal_and_txt(txt_file_path, terminal_output):
    if not txt_file_path or not terminal_output:
        return 0

    # read txt file content
    with open(txt_file_path, 'r') as f:
        txt_file_content = f.read()
    # compare terminal output with txt file content
    return 1 if terminal_output == txt_file_content else 0


def compare_python_pure_text(py_file_path, gold_file_path):
    if not py_file_path or not gold_file_path:
        return 0

    # first, change the suffix of gold_file from .txt to .py
    print("py_file_path: ")
    print(py_file_path)
    print("gold_file_path: ")
    print(gold_file_path)

    # gold_file_path = gold_file_path.replace('.txt', '.py')
    def remove_whitespace(text):
        return ''.join(text.split())

    with open(py_file_path, 'r') as file1:
        content1 = file1.read()
    with open(gold_file_path, 'r') as file2:
        content2 = file2.read()
    content1_no_whitespace = remove_whitespace(content1)
    content2_no_whitespace = remove_whitespace(content2)
    if content1_no_whitespace == content2_no_whitespace:
        return 1
    else:
        return 0


def check_step_actions(result, rules):
    # TODO: Implement me
    
    def point_is_in_bbox(point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    expected = rules["expected"]    
    cur_action_idx = 0
    cur_pos = (-1, -1)
    for action in result:
        if action == "FAIL":
            return 0.
        elif action in ["WAIT", "DONE"]:
            continue
        action_type = action["action_type"].upper()
        params = action["parameters"] if "parameters" in action else {param: action[param] for param in action if param != 'action_type'}
        if cur_action_idx >= len(expected):
            return 0.
        gt_action = expected[cur_action_idx]
        gt_params = gt_action["parameters"]
        if action_type == "MOVE_TO":
            cur_pos = (params["x"], params["y"])
            if gt_action["action_type"] == "MOVE_TO":
                if not point_is_in_bbox(cur_pos, gt_params["bbox"]):
                    return 0.
            continue
        
        if gt_action["action_type"] != action_type:
            return 0.
        if action_type in ["CLICK", "RIGHT_CLICK", "DOUBLE_CLICK", "DRAG_TO"]:
            if "x" in params and "y" in params:
                cur_pos = (params["x"], params["y"])
            if not point_is_in_bbox(cur_pos, gt_params["bbox"]):
                return 0.
        elif action_type in ["SCROLL", "MOUSE_DOWN", "MOUSE_UP"]:
            # TODO: how to deal with these?
            pass
        elif action_type == "TYPING":
            if params["text"] != gt_params["text"]:
                return 0.
        elif action_type in ["PRESS", "KEY_DOWN", "KEY_UP"]:
            if params["key"] != gt_params["key"]:
                return 0.
        elif action_type == "HOTKEY":
            if params["keys"] != gt_params["keys"]:
                return 0.
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
        cur_action_idx += 1
        
    return 1.


def compare_files_exactly(result_path, expect_path):
    return float(filecmp.cmp(result_path, expect_path, shallow=False))


def compare_folders_exactly(result_path, expect_path):
    result = subprocess.run(["diff", "-rq", result_path, expect_path], capture_output=True, text=True)
    return float(result.returncode == 0)


def compare_mail_content(result, expected, **options):            
    if expected.get("to", result["to"]) != result["to"]:
        return 0.0
    if expected.get("subject", result["subject"]) != result["subject"]:
        return 0.0
    if "body" in expected:
        body_type = options.get("body_type", "contains")
        if options.get("ignore_blanks", False):
            result["body"] = re.sub(r'[\t\n]', ' ', result["body"]).strip()
            result["body"] = re.sub(r'\s+', ' ', result["body"])
            expected["body"] = re.sub(r'[\t\n]', ' ', expected["body"]).strip()
            expected["body"] = re.sub(r'\s+', ' ', expected["body"])
        if body_type == "contains" and not expected["body"] in result["body"]:
            return 0.0
        if body_type == "equals" and not expected["body"] == result["body"]:
            return 0.0
    if "attachments" in expected:
        if len(result["attachments"]) != len(expected["attachments"]):
            return 0.0
        for result_file_path, expected_file_path in zip(sorted(result["attachments"]), sorted(expected["attachments"])):
            if result_file_path != expected_file_path.replace("_gt", ""):
                return 0.0
            if result_file_path.endswith(".pdf"):
                if compare_pdfs(result_file_path, expected_file_path) < 0.5:
                    return 0.0
            else:
                if not compare_files_exactly(result_file_path, expected_file_path):
                    return 0.0
    return 1.0


def check_2048_has_16(result):
    return float(any(int(item) >= 16 for item in result["tiles"]))


def check_email_match_href(result, rules):
    expected = rules["expected"]
    if expected.startswith("mailto:"):
        expected = expected[7:]
    return float(result == expected)


def is_not_empty(result):
    return float(bool(result))


if __name__ == '__main__':
    print(check_direct_json_object([], rules={
                "relativeTime": {
                  "from": "5th next month"
                },
                "expected": {
                    "start": "SEA",
                    "end": "NYC",
                    "time": "{DoW}, {Month} {DayD}, {Year}",
                    "category": "Miles"
                }}))