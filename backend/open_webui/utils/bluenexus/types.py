"""
Type definitions for BlueNexus Data Client
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypeVar, Generic
from pydantic import BaseModel, Field


class SortOrder(str, Enum):
    """Sort order for query results"""
    ASC = "asc"
    DESC = "desc"


class SortBy(str, Enum):
    """Fields available for sorting"""
    CREATED_AT = "createdAt"
    UPDATED_AT = "updatedAt"


class QueryOptions(BaseModel):
    """Options for querying records from BlueNexus"""
    filter: Optional[dict[str, Any]] = None
    sort_by: SortBy = SortBy.CREATED_AT
    sort_order: SortOrder = SortOrder.DESC
    limit: int = Field(default=20, ge=1, le=100)
    page: int = Field(default=1, ge=1)

    class Config:
        use_enum_values = True


class PaginationInfo(BaseModel):
    """Pagination metadata from BlueNexus response"""
    page: int
    limit: int
    total: int
    pages: int
    hasNext: bool
    hasPrev: bool


class BlueNexusRecord(BaseModel):
    """
    Base record model from BlueNexus User-Data API.

    All records have these system fields plus any custom fields.
    """
    id: str
    createdAt: datetime
    updatedAt: datetime
    schemaUri: Optional[str] = None

    # Allow extra fields for flexible schema
    class Config:
        extra = "allow"

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary including extra fields"""
        return self.model_dump(mode="json")

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "BlueNexusRecord":
        """Create a record from API response data"""
        return cls(**data)


T = TypeVar("T", bound=BlueNexusRecord)


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response from BlueNexus query endpoint"""
    data: list[dict[str, Any]]  # Raw data, convert to records as needed
    pagination: PaginationInfo

    def get_records(self, record_class: type[T] = BlueNexusRecord) -> list[T]:
        """Convert raw data to typed record objects"""
        return [record_class(**item) for item in self.data]


class ValidationError(BaseModel):
    """Validation error from schema validation"""
    field: str
    message: str
    value: Optional[Any] = None


class VerifyResponse(BaseModel):
    """Response from data verification endpoint"""
    valid: bool
    errors: Optional[list[ValidationError]] = None


class BlueNexusError(Exception):
    """Base exception for BlueNexus API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class BlueNexusAuthError(BlueNexusError):
    """Authentication/authorization error (401/403)"""
    pass


class BlueNexusNotFoundError(BlueNexusError):
    """Resource not found error (404)"""
    pass


class BlueNexusValidationError(BlueNexusError):
    """Validation error (400)"""
    def __init__(self, message: str, errors: Optional[list[ValidationError]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.errors = errors or []


class BlueNexusConnectionError(BlueNexusError):
    """Connection/network error"""
    pass
