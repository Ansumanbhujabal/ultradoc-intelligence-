"""Shipment data extraction schema with robust post-processing validators."""

import re
from pydantic import BaseModel, Field, field_validator
from typing import Optional


# Values that should be treated as null
NULL_VALUES = {"", "-", "--", "---", "n/a", "na", "none", "null", "tbd", "unknown", "not available", "not found"}


def _normalize_null(v):
    """Convert dash-like, N/A, and empty values to None."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in NULL_VALUES:
        return None
    return v


def _clean_rate(v):
    """Strip currency symbols, commas, whitespace from rate values."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in NULL_VALUES:
            return None
        # Remove currency symbols and codes
        cleaned = re.sub(r"[$$€£¥₹,]", "", v)
        cleaned = re.sub(r"\b(USD|CAD|EUR|MXN|GBP|INR)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return v


def _normalize_datetime(v):
    """Attempt to normalize common date formats to ISO-ish string."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in NULL_VALUES:
            return None
        # Already ISO format
        if re.match(r"\d{4}-\d{2}-\d{2}", v):
            return v
        # Try common formats
        import datetime
        formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%d-%b-%Y", "%B %d, %Y",
            "%m.%d.%Y", "%d/%m/%Y", "%Y.%m.%d",
            "%m/%d/%Y %H:%M", "%m-%d-%Y %H:%M",
            "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(v, fmt)
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
        # Return as-is if no format matched — better than losing data
        return v
    return v


class ShipmentData(BaseModel):
    """Structured shipment data extracted from logistics documents."""

    shipment_id: Optional[str] = Field(None, description="Load ID or shipment reference number")
    shipper: Optional[str] = Field(None, description="Shipper name and address")
    consignee: Optional[str] = Field(None, description="Consignee/receiver name and address")
    pickup_datetime: Optional[str] = Field(None, description="Pickup date and time in ISO format")
    delivery_datetime: Optional[str] = Field(None, description="Delivery date and time in ISO format")
    equipment_type: Optional[str] = Field(None, description="Equipment type (e.g., Flatbed, Dry Van)")
    mode: Optional[str] = Field(None, description="Shipping mode (e.g., FTL, LTL)")
    rate: Optional[float] = Field(None, description="Rate/charge amount as a number")
    currency: Optional[str] = Field(None, description="Currency code (e.g., USD)")
    weight: Optional[str] = Field(None, description="Weight with unit (e.g., '56000 lbs')")
    carrier_name: Optional[str] = Field(None, description="Carrier company name")

    @field_validator("shipment_id", "shipper", "consignee", "equipment_type", "mode", "currency", "weight", "carrier_name", mode="before")
    @classmethod
    def normalize_string_fields(cls, v):
        return _normalize_null(v)

    @field_validator("rate", mode="before")
    @classmethod
    def clean_rate_field(cls, v):
        return _clean_rate(v)

    @field_validator("pickup_datetime", "delivery_datetime", mode="before")
    @classmethod
    def normalize_datetime_fields(cls, v):
        return _normalize_datetime(v)

    def completeness_score(self) -> float:
        fields = self.model_fields.keys()
        non_null = sum(1 for f in fields if getattr(self, f) is not None)
        return round(non_null / len(fields), 2)
