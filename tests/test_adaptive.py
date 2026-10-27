"""M7 路由单测：各问题特征命中正确策略。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.query_augment import adaptive_strategy


def test_tf_stays_original():
    assert adaptive_strategy("One study reports X. True or False: the findings contradict each other.") == "original"


def test_numeric_gets_expansion():
    assert adaptive_strategy("How many petajoules of electricity were reported for training?") == "expansion"


def test_long_gets_abstraction():
    q = "According to a multi-year analysis across several reporting frameworks and boundaries, " * 2 + "what did it find?"
    assert adaptive_strategy(q) == "abstraction"


def test_definitional_gets_paraphrase():
    assert adaptive_strategy("What is the term for water evaporated during power generation?") == "paraphrase"
