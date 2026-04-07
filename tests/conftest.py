"""Shared test fixtures."""
import os
import pytest

SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ultradoc_sample_test_data"
)

@pytest.fixture
def sample_bol_path():
    return os.path.join(SAMPLE_DATA_DIR, "BOL53657_billoflading.pdf")

@pytest.fixture
def sample_carrier_rc_path():
    return os.path.join(SAMPLE_DATA_DIR, "LD53657-Carrier-RC.pdf")

@pytest.fixture
def sample_shipper_rc_path():
    return os.path.join(SAMPLE_DATA_DIR, "LD53657-Shipper-RC.pdf")
