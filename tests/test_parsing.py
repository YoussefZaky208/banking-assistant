"""Unit tests for reading the LLM's JSON and the language detector."""
from src.rag.knowledge_base import detect_lang
from src.rag.rewriter import parse_analysis


def test_reads_valid_json():
    out = parse_analysis('{"route": "handoff", "query": "", "upset": true, "injection": false}', "x")
    assert out == {"route": "handoff", "query": "", "upset": True, "injection": False}


def test_json_inside_extra_text():
    out = parse_analysis('Sure! {"route": "chitchat", "query": ""}', "hi")
    assert out["route"] == "chitchat"


def test_broken_json_falls_back_to_question():
    out = parse_analysis("this is not json", "ATM fee?")
    assert out["route"] == "question" and out["query"] == "ATM fee?"


def test_unknown_route_falls_back_to_question():
    assert parse_analysis('{"route": "dance"}', "x")["route"] == "question"


def test_language_detection():
    assert detect_lang("فقدت بطاقتي") == "ar"
    assert detect_lang("I lost my card") == "en"
    assert detect_lang("بطاقتي فيها مشكلة CVV") == "ar"
