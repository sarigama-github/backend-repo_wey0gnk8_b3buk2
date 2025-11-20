import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Order, AdSpend, SubscriptionEvent, COGS

app = FastAPI(title="E-commerce Profit Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Profit Tracker API running"}


# ----- Ingestion Endpoints -----
@app.post("/api/orders")
def create_order(order: Order):
    order_dict = order.model_dump()
    order_dict["date"] = order.date.replace(tzinfo=timezone.utc)
    _id = create_document("order", order_dict)
    return {"inserted_id": _id}


@app.post("/api/adspend")
def create_ad_spend(spend: AdSpend):
    doc = spend.model_dump()
    doc["date"] = spend.date.replace(tzinfo=timezone.utc)
    _id = create_document("adspend", doc)
    return {"inserted_id": _id}


@app.post("/api/subscription-events")
def create_subscription_event(evt: SubscriptionEvent):
    doc = evt.model_dump()
    doc["date"] = evt.date.replace(tzinfo=timezone.utc)
    _id = create_document("subscriptionevent", doc)
    return {"inserted_id": _id}


@app.post("/api/cogs")
def create_cogs(cogs: COGS):
    doc = cogs.model_dump()
    doc["effective_date"] = cogs.effective_date.replace(tzinfo=timezone.utc)
    _id = create_document("cogs", doc)
    return {"inserted_id": _id}


# ----- Metrics Helpers -----

def daterange(start_date: datetime, end_date: datetime):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)


def start_of_day(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def end_of_day(dt: datetime) -> datetime:
    sod = start_of_day(dt)
    return sod + timedelta(days=1) - timedelta(microseconds=1)


def aggregate_daily_metrics(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    # Build base map for all days
    days = {}
    for d in daterange(start_of_day(start_date), start_of_day(end_date)):
        key = d.strftime("%Y-%m-%d")
        days[key] = {
            "date": key,
            "revenue": 0.0,
            "orders": 0,
            "ad_spend": 0.0,
            "cogs": 0.0,
            "processing_fees": 0.0,
            "profit": 0.0,
        }

    # Orders
    orders = list(db["order"].find({
        "date": {"$gte": start_of_day(start_date), "$lte": end_of_day(end_date)}
    })) if db else []

    for o in orders:
        day = o["date"].astimezone(timezone.utc).strftime("%Y-%m-%d")
        revenue_net = float(o.get("subtotal", 0)) - float(o.get("discounts", 0)) - float(o.get("refunds", 0)) + float(o.get("shipping_revenue", 0))
        fees = float(o.get("processing_fees", 0))
        cogs_sum = 0.0
        for li in o.get("line_items", []) or []:
            qty = int(li.get("qty", 0))
            cost = float(li.get("cost", 0))
            cogs_sum += qty * cost
        days[day]["revenue"] += max(revenue_net, 0.0)
        days[day]["orders"] += 1
        days[day]["cogs"] += cogs_sum
        days[day]["processing_fees"] += fees

    # Ad Spend
    spends = list(db["adspend"].find({
        "date": {"$gte": start_of_day(start_date), "$lte": end_of_day(end_date)}
    })) if db else []
    for s in spends:
        day = s["date"].astimezone(timezone.utc).strftime("%Y-%m-%d")
        days[day]["ad_spend"] += float(s.get("amount", 0))

    # Profit
    for k, v in days.items():
        v["profit"] = v["revenue"] - v["cogs"] - v["processing_fees"] - v["ad_spend"]

    # Return in order
    out = [days[d.strftime("%Y-%m-%d")] for d in daterange(start_of_day(start_date), start_of_day(end_date))]
    return out


def compute_daily_mrr(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Compute daily MRR run-rate using subscription events as deltas."""
    days = {}
    for d in daterange(start_of_day(start_date), start_of_day(end_date)):
        key = d.strftime("%Y-%m-%d")
        days[key] = {"date": key, "mrr": 0.0, "net_new_mrr": 0.0}

    events = list(db["subscriptionevent"].find({
        "date": {"$lte": end_of_day(end_date)}
    })) if db else []

    # Sort events by date
    events.sort(key=lambda e: e["date"])  # UTC aware

    # Accumulate MRR over time
    mrr = 0.0
    day_keys = [d.strftime("%Y-%m-%d") for d in daterange(start_of_day(start_date), start_of_day(end_date))]

    idx = 0
    for d in day_keys:
        day_dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        net_new_today = 0.0
        while idx < len(events) and events[idx]["date"] <= end_of_day(day_dt):
            amt = float(events[idx].get("amount", 0))
            etype = events[idx].get("event_type")
            # churn reduces MRR (amount should be positive; we'll subtract)
            if etype == "churn":
                net_new_today -= amt
                mrr -= amt
            elif etype == "contraction":
                net_new_today -= amt
                mrr -= amt
            else:
                net_new_today += amt
                mrr += amt
            idx += 1
        days[d]["net_new_mrr"] = net_new_today
        days[d]["mrr"] = max(mrr, 0.0)

    return [days[k] for k in day_keys]


# ----- Public Metrics Endpoints -----
@app.get("/api/daily-summary")
def daily_summary(
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """Daily revenue, ad spend, COGS, processing fees, profit"""
    today = datetime.now(timezone.utc)
    end_date = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end else today
    start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start else end_date - timedelta(days=6)

    data = aggregate_daily_metrics(start_date, end_date)
    totals = {
        "revenue": round(sum(d["revenue"] for d in data), 2),
        "orders": sum(d["orders"] for d in data),
        "ad_spend": round(sum(d["ad_spend"] for d in data), 2),
        "cogs": round(sum(d["cogs"] for d in data), 2),
        "processing_fees": round(sum(d["processing_fees"] for d in data), 2),
        "profit": round(sum(d["profit"] for d in data), 2),
    }
    return {"range": {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")}, "days": data, "totals": totals}


@app.get("/api/mrr-forecast")
def mrr_forecast(days_ahead: int = 60):
    """Forecast MRR using average net-new MRR from last 30 days as simple trend."""
    today = datetime.now(timezone.utc)
    lookback_start = today - timedelta(days=60)

    hist = compute_daily_mrr(lookback_start, today)
    if len(hist) == 0:
        return {"today_mrr": 0.0, "daily_net_new_avg": 0.0, "forecast": []}

    # Use last 30 days net new MRR average
    recent = hist[-30:] if len(hist) >= 30 else hist
    avg_net_new = sum(h["net_new_mrr"] for h in recent) / len(recent)
    current_mrr = hist[-1]["mrr"]

    forecast = []
    mrr = current_mrr
    for i in range(1, days_ahead + 1):
        mrr = max(mrr + avg_net_new, 0.0)
        day = (today + timedelta(days=i)).astimezone(timezone.utc).strftime("%Y-%m-%d")
        forecast.append({"date": day, "mrr": round(mrr, 2)})

    return {
        "today": today.strftime("%Y-%m-%d"),
        "today_mrr": round(current_mrr, 2),
        "daily_net_new_avg": round(avg_net_new, 2),
        "forecast": forecast,
        "history": hist[-30:],
    }


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available" if db is None else "✅ Connected",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "collections": []
    }
    try:
        if db is not None:
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:100]}"
    return response


# ----- Seed demo data -----
class SeedConfig(BaseModel):
    days: int = 30
    base_revenue: float = 3000.0
    base_orders: int = 50
    base_cogs_rate: float = 0.35  # percent of revenue
    base_ad_spend: float = 1200.0
    base_fees_rate: float = 0.03
    start_mrr: float = 10000.0
    avg_net_new_mrr_per_day: float = 50.0


@app.post("/api/seed-demo")
def seed_demo(cfg: SeedConfig):
    if db is None:
        return {"error": "Database not configured"}

    start_day = start_of_day(datetime.now(timezone.utc) - timedelta(days=cfg.days - 1))

    # Clear existing demo docs (optional)
    db["order"].delete_many({"date": {"$gte": start_day}})
    db["adspend"].delete_many({"date": {"$gte": start_day}})
    db["subscriptionevent"].delete_many({"date": {"$gte": start_day}})

    import random

    # Seed orders, cogs derived from line items
    for i, d in enumerate(daterange(start_day, datetime.now(timezone.utc))):
        # Randomize around base values
        rev = max(0, random.gauss(cfg.base_revenue, cfg.base_revenue * 0.15))
        orders = max(1, int(random.gauss(cfg.base_orders, cfg.base_orders * 0.2)))
        discounts = rev * random.uniform(0.05, 0.15)
        refunds = rev * random.uniform(0.00, 0.05)
        ship_rev = rev * random.uniform(0.02, 0.06)
        fees = rev * cfg.base_fees_rate
        # derive line items
        avg_items_per_order = random.uniform(1.1, 1.8)
        items = int(orders * avg_items_per_order)
        cogs_rate = random.uniform(cfg.base_cogs_rate * 0.9, cfg.base_cogs_rate * 1.1)
        # create one order per 10 orders to keep doc count reasonable
        chunk = max(1, orders // 10)
        for j in range(chunk):
            li_count = max(1, items // chunk)
            line_items = []
            for k in range(li_count):
                price = max(5.0, rev / items)
                unit_cost = price * cogs_rate * random.uniform(0.9, 1.1)
                line_items.append({
                    "sku": f"SKU-{k%8}",
                    "title": f"Product {k%8}",
                    "qty": 1,
                    "price": round(price, 2),
                    "cost": round(unit_cost, 2)
                })
            order_doc = {
                "order_id": f"D{i}-O{j}",
                "date": d,
                "subtotal": round(rev, 2),
                "discounts": round(discounts, 2),
                "refunds": round(refunds, 2),
                "shipping_revenue": round(ship_rev, 2),
                "processing_fees": round(fees, 2),
                "line_items": line_items,
            }
            db["order"].insert_one(order_doc)

        # Ad spend
        ad_spend = max(0, random.gauss(cfg.base_ad_spend, cfg.base_ad_spend * 0.2))
        db["adspend"].insert_many([
            {"date": d, "channel": "meta", "amount": round(ad_spend * 0.6, 2), "kind": "cold"},
            {"date": d, "channel": "google", "amount": round(ad_spend * 0.4, 2), "kind": "warm"},
        ])

    # Seed subscription events to reach start_mrr then add daily net new
    # Establish starting MRR at first day
    first_day = start_day
    db["subscriptionevent"].insert_one({
        "date": first_day,
        "amount": float(cfg.start_mrr),
        "event_type": "reactivation"
    })
    for d in daterange(first_day, datetime.now(timezone.utc)):
        delta = random.gauss(cfg.avg_net_new_mrr_per_day, abs(cfg.avg_net_new_mrr_per_day) * 0.5)
        # Randomly split delta into components
        if delta >= 0:
            db["subscriptionevent"].insert_one({"date": d, "amount": round(abs(delta), 2), "event_type": "new"})
        else:
            db["subscriptionevent"].insert_one({"date": d, "amount": round(abs(delta), 2), "event_type": "churn"})

    return {"status": "ok", "seeded_days": cfg.days}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
