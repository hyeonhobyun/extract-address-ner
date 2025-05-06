from pydantic import BaseModel
from typing import List, Optional


class AddressRequest(BaseModel):
    text: str


class Address(BaseModel):
    text: str
    start: int
    end: int
    confidence: float


class AddressResponse(BaseModel):
    addresses: List[Address]
    original_text: str


class AddressValidationRequest(BaseModel):
    address: str


class AddressValidationResponse(BaseModel):
    address: str
    is_valid: bool
    confidence: float


class TrainingData(BaseModel):
    text: str
    address: str
    is_valid: bool


class TrainingResponse(BaseModel):
    status: str
    message: str
