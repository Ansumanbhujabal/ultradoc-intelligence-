"""Tests for document parser."""
import pytest
from app.pipeline.parser import parse_document

def test_parse_pdf(sample_bol_path):
    result = parse_document(sample_bol_path)
    assert result["status"] == "success"
    assert result["page_count"] >= 1
    assert len(result["text"]) > 100
    assert "LD53657" in result["text"]

def test_parse_pdf_extracts_key_fields(sample_carrier_rc_path):
    result = parse_document(sample_carrier_rc_path)
    assert "SWIFT SHIFT LOGISTICS" in result["text"]
    assert "400" in result["text"]

def test_parse_txt(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Shipment ID: LD99999\nCarrier: Test Carrier")
    result = parse_document(str(txt_file))
    assert result["status"] == "success"
    assert "LD99999" in result["text"]
    assert result["page_count"] == 1

def test_parse_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("data")
    result = parse_document(str(bad_file))
    assert result["status"] == "error"
