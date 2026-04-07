"""Shipment data extraction schema for structured output."""

from pydantic import BaseModel, Field
from typing import Optional


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

    def completeness_score(self) -> float:
        fields = self.model_fields.keys()
        non_null = sum(1 for f in fields if getattr(self, f) is not None)
        return round(non_null / len(fields), 2)
