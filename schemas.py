"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

# Example schemas (keep for reference)
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# ---- App Schemas ----

class OrderLineItem(BaseModel):
    sku: str
    title: str
    qty: int = Field(..., ge=1)
    price: float = Field(..., ge=0, description="Unit price paid by customer")
    cost: float = Field(..., ge=0, description="Unit cost of goods")

class Order(BaseModel):
    order_id: str
    date: datetime
    subtotal: float = Field(..., ge=0)
    discounts: float = Field(0, ge=0)
    refunds: float = Field(0, ge=0)
    shipping_revenue: float = Field(0, ge=0)
    processing_fees: float = Field(0, ge=0)
    line_items: List[OrderLineItem]

class AdSpend(BaseModel):
    date: datetime
    channel: str
    amount: float = Field(..., ge=0)
    kind: Literal["cold", "warm", "brand", "other"] = "cold"

class SubscriptionEvent(BaseModel):
    date: datetime
    amount: float = Field(..., ge=0, description="MRR amount delta for this event")
    event_type: Literal["new", "expansion", "contraction", "churn", "reactivation"]

class COGS(BaseModel):
    sku: str
    cost: float = Field(..., ge=0)
    effective_date: datetime

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!
