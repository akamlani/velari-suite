from    enum import StrEnum, auto
from    pydantic import BaseModel, Field
# package modules
from    ..types import Priority, Sentiment


class TicketCategory(StrEnum):
    BILLING     = auto()
    SHIPPING    = auto()
    RETURNS     = auto()
    PRODUCT     = auto()
    TECHNICAL   = auto()
    GENERAL     = auto()


class Ticket(BaseModel):
    """Structured customer support ticket extracted from a free-text message."""
    category:   TicketCategory  = Field(description="Best-fit category for the ticket.")
    priority:   Priority        = Field(description="Urgency, based on customer impact and time sensitivity.")
    sentiment:  Sentiment       = Field(description="Overall tone of the customer's message.")
    summary:    str             = Field(description="One-sentence summary of the customer's issue.")
