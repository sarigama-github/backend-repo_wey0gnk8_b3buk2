import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

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

MARKETS = [
    "USA",
    "UK",
    "Australia",
    "New Zealand",
    "Singapore",
    "Canada",
    "Ireland",
]


@app.get("/")
def read_root():
    return {"message": "Profit Tracker API running"}


@app.get("/api/markets")
def list_markets():
    """Return known markets. If DB has distinct markets, use them; otherwise default list."""
    try:
        markets = set()
        if db is not None:
            for col in ["order", "adspend", "subscriptionevent"]:
                try:
                    markets.update([m for m in db[col].distinct("market") if m])
                except Exception:
                    pass
        if markets:
            return sorted(markets)
    except Exception:
        pass
    return MARKETS


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
    cur = start_of_day(start_date)
    end_date = start_of_day(end_date)
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)


def start_of_day(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def end_of_day(dt: datetime) -> datetime:
    sod = start_of_day(dt)
    return sod + timedelta(days=1) - timedelta(microseconds=1)


def aggregate_daily_metrics(start_date: datetime, end_date: datetime, market: Optional[str] = None) -> List[Dict[str, Any]]:
    # Build base map for all days
    days: Dict[str, Dict[str, Any]] = {}
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

    query_base = {"date": {"$gte": start_of_day(start_date), "$lte": end_of_day(end_date)}}
    if market:
        query_base["market"] = market

    # Orders
    orders = list(db["order"].find(query_base)) if db else []

    for o in orders:
        day = o["date"].astimezone(timezone.utc).strftime("%Y-%m-%d")
        revenue_net = float(o.get("subtotal", 0)) - float(o.get("discounts", 0)) - float(o.get("refunds", 0)) + float(o.get("shipping_revenue", 0))
        fees = float(o.get("processing_fees", 0))
        cogs_sum = 0.0
        for li in o.get("line_items", []) or []:
            qty = int(li.get("qty", 0))
            cost = float(li.get("cost", 0))
            cogs_sum += qty * cost
        if day in days:
            days[day]["revenue"] += max(revenue_net, 0.0)
            days[day]["orders"] += 1
            days[day]["cogs"] += cogs_sum
            days[day]["processing_fees"] += fees

    # Ad Spend
    spends = list(db["adspend"].find(query_base)) if db else []
    for s in spends:
        day = s["date"].astimezone(timezone.utc).strftime("%Y-%m-%d")
        if day in days:
            days[day]["ad_spend"] += float(s.get("amount", 0))

    # Profit
    for v in days.values():
        v["profit"] = v["revenue"] - v["cogs"] - v["processing_fees"] - v["ad_spend"]

    # Return in order
    out = [days[d.strftime("%Y-%m-%d")] for d in daterange(start_of_day(start_date), start_of_day(end_date))]
    return out


def compute_daily_mrr(start_date: datetime, end_date: datetime, market: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compute daily MRR run-rate using subscription events as deltas."""
    days: Dict[str, Dict[str, Any]] = {}
    for d in daterange(start_of_day(start_date), start_of_day(end_date)):
        key = d.strftime("%Y-%m-%d")
        days[key] = {"date": key, "mrr": 0.0, "net_new_mrr": 0.0}

    query = {"date": {"$lte": end_of_day(end_date)}}
    if market:
        query["market"] = market

    events = list(db["subscriptionevent"].find(query)) if db else []

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
    market: Optional[str] = Query(None, description="Filter by market"),
):
    """Daily revenue, ad spend, COGS, processing fees, profit"""
    today = datetime.now(timezone.utc)
    end_date = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end else today
    start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start else end_date - timedelta(days=6)

    data = aggregate_daily_metrics(start_date, end_date, market)
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
def mrr_forecast(days_ahead: int = 60, market: Optional[str] = Query(None, description="Filter by market")):
    """Forecast MRR using average net-new MRR from last 30 days as simple trend."""
    today = datetime.now(timezone.utc)
    lookback_start = today - timedelta(days=60)

    hist = compute_daily_mrr(lookback_start, today, market)
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

    # Seed per-market data for each day
    for i, d in enumerate(daterange(start_day, datetime.now(timezone.utc))):
        for m in MARKETS:
            # Randomize around base values, with small market factor
            market_factor = random.uniform(0.7, 1.2)
            rev = max(0, random.gauss(cfg.base_revenue, cfg.base_revenue * 0.15)) * market_factor
            orders = max(1, int(random.gauss(cfg.base_orders, cfg.base_orders * 0.2) * market_factor))
            discounts = rev * random.uniform(0.05, 0.15)
            refunds = rev * random.uniform(0.00, 0.05)
            ship_rev = rev * random.uniform(0.02, 0.06)
            fees = rev * cfg.base_fees_rate
            # derive line items
            avg_items_per_order = random.uniform(1.1, 1.8)
            items = max(1, int(orders * avg_items_per_order))
            cogs_rate = random.uniform(cfg.base_cogs_rate * 0.9, cfg.base_cogs_rate * 1.1)
            # create one order doc per ~10 orders to keep doc count reasonable
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
                    "order_id": f"{m[:2].upper()}-{i}-O{j}",
                    "date": d,
                    "subtotal": round(rev, 2),
                    "discounts": round(discounts, 2),
                    "refunds": round(refunds, 2),
                    "shipping_revenue": round(ship_rev, 2),
                    "processing_fees": round(fees, 2),
                    "line_items": line_items,
                    "market": m,
                }
                db["order"].insert_one(order_doc)

            # Ad spend per market
            ad_spend = max(0, random.gauss(cfg.base_ad_spend, cfg.base_ad_spend * 0.2)) * market_factor
            db["adspend"].insert_many([
                {"date": d, "channel": "meta", "amount": round(ad_spend * 0.6, 2), "kind": "cold", "market": m},
                {"date": d, "channel": "google", "amount": round(ad_spend * 0.4, 2), "kind": "warm", "market": m},
            ])

            # Subscription events per market to reach start_mrr proportionally
            if i == 0:
                start_share = cfg.start_mrr / len(MARKETS)
                db["subscriptionevent"].insert_one({
                    "date": d,
                    "amount": float(start_share),
                    "event_type": "reactivation",
                    "market": m,
                })
            delta = random.gauss(cfg.avg_net_new_mrr_per_day, abs(cfg.avg_net_new_mrr_per_day) * 0.5) * random.uniform(0.6, 1.4)
            if delta >= 0:
                db["subscriptionevent"].insert_one({"date": d, "amount": round(abs(delta), 2), "event_type": "new", "market": m})
            else:
                db["subscriptionevent"].insert_one({"date": d, "amount": round(abs(delta), 2), "event_type": "churn", "market": m})

    return {"status": "ok", "seeded_days": cfg.days, "markets": MARKETS}


# ----- Observations (AI CFO) -----
class ObservationsResponse(BaseModel):
    range: Dict[str, str]
    market: Optional[str] = None
    observations: List[Dict[str, Any]]
    facebook: List[Dict[str, Any]]
    shopify: List[Dict[str, Any]]
    per_market: Dict[str, Dict[str, Any]]


def _sum_days(days: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "revenue": sum(d.get("revenue", 0.0) for d in days),
        "orders": sum(d.get("orders", 0) for d in days),
        "ad_spend": sum(d.get("ad_spend", 0.0) for d in days),
        "cogs": sum(d.get("cogs", 0.0) for d in days),
        "processing_fees": sum(d.get("processing_fees", 0.0) for d in days),
        "profit": sum(d.get("profit", 0.0) for d in days),
    }


def _calc_ratios(t: Dict[str, float], meta_spend: float = 0.0) -> Dict[str, float]:
    revenue = t.get("revenue", 0.0) or 0.0
    orders = t.get("orders", 0) or 0
    ad_spend = t.get("ad_spend", 0.0) or 0.0
    cogs = t.get("cogs", 0.0) or 0.0
    fees = t.get("processing_fees", 0.0) or 0.0
    ratios = {
        "aov": (revenue / orders) if orders else 0.0,
        "gross_margin_pct": ((revenue - cogs) / revenue * 100.0) if revenue else 0.0,
        "fee_pct": (fees / revenue * 100.0) if revenue else 0.0,
        "mer": (revenue / ad_spend) if ad_spend else 0.0,  # blended ROAS/MER
        "cpa": (ad_spend / orders) if orders else 0.0,
        "cpa_meta": (meta_spend / orders) if orders else 0.0,
    }
    return ratios


def _ad_spend_breakdown(start_date: datetime, end_date: datetime, market: Optional[str]) -> Dict[str, float]:
    query = {"date": {"$gte": start_of_day(start_date), "$lte": end_of_day(end_date)}}
    if market:
        query["market"] = market
    spends = list(db["adspend"].find(query)) if db else []
    out: Dict[str, float] = {}
    for s in spends:
        ch = (s.get("channel") or "other").lower()
        out[ch] = out.get(ch, 0.0) + float(s.get("amount", 0))
    return out


@app.get("/api/observations")
def observations(
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
    market: Optional[str] = Query(None, description="Filter by market"),
):
    """AI CFO observations across Facebook and Shopify, per market and global, for n8n integration."""
    today = datetime.now(timezone.utc)
    end_date = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end else today
    start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start else end_date - timedelta(days=6)

    # Per-market rollups
    markets = list_markets()
    per_market: Dict[str, Dict[str, Any]] = {}
    for mkt in markets:
        days = aggregate_daily_metrics(start_date, end_date, mkt)
        totals = _sum_days(days)
        ch = _ad_spend_breakdown(start_date, end_date, mkt)
        meta_spend = ch.get("meta", 0.0)
        ratios = _calc_ratios(totals, meta_spend)
        per_market[mkt] = {**totals, **ratios, "meta_spend": meta_spend, "channel_spend": ch}

    # Global/selected market
    if market:
        sel_days = aggregate_daily_metrics(start_date, end_date, market)
        sel_totals = _sum_days(sel_days)
        sel_channels = _ad_spend_breakdown(start_date, end_date, market)
        sel_meta = sel_channels.get("meta", 0.0)
        sel_ratios = _calc_ratios(sel_totals, sel_meta)
        current = {**sel_totals, **sel_ratios, "meta_spend": sel_meta, "channel_spend": sel_channels}
    else:
        # Global totals across all markets
        glob_days = aggregate_daily_metrics(start_date, end_date, None)
        glob_totals = _sum_days(glob_days)
        glob_channels = _ad_spend_breakdown(start_date, end_date, None)
        glob_meta = glob_channels.get("meta", 0.0)
        glob_ratios = _calc_ratios(glob_totals, glob_meta)
        current = {**glob_totals, **glob_ratios, "meta_spend": glob_meta, "channel_spend": glob_channels}

    # Rankings
    by_revenue = sorted(per_market.items(), key=lambda kv: kv[1].get("revenue", 0.0), reverse=True)
    by_profit = sorted(per_market.items(), key=lambda kv: kv[1].get("profit", 0.0), reverse=True)
    top_rev_market, top_rev_val = (by_revenue[0][0], by_revenue[0][1]["revenue"]) if by_revenue else (None, 0.0)
    top_profit_market, top_profit_val = (by_profit[0][0], by_profit[0][1]["profit"]) if by_profit else (None, 0.0)

    # Benchmarks
    AOV_TARGET = 70.0
    GM_TARGET = 80.0
    FEES_TARGET = 5.0
    CPA_META_TARGET = 50.0

    # Build Shopify (e-commerce) observations
    shopify_obs: List[Dict[str, Any]] = []
    if top_rev_market:
        shopify_obs.append({
            "type": "revenue_leader",
            "severity": "info",
            "message": f"{top_rev_market} is the top revenue market (${top_rev_val:,.0f}). Consider prioritizing inventory and ops here.",
            "market": top_rev_market,
        })
    if top_profit_market:
        shopify_obs.append({
            "type": "profit_leader",
            "severity": "info",
            "message": f"{top_profit_market} leads profit (${top_profit_val:,.0f}). Allocate scale budget where margin is strongest.",
            "market": top_profit_market,
        })

    # Compare AOV and margin against global average
    if per_market:
        avg_aov = sum(v.get("aov", 0.0) for v in per_market.values()) / max(len(per_market), 1)
        avg_gm = sum(v.get("gross_margin_pct", 0.0) for v in per_market.values()) / max(len(per_market), 1)
        for mkt, vs in per_market.items():
            if vs.get("aov", 0) >= avg_aov * 1.15:
                shopify_obs.append({
                    "type": "high_aov",
                    "severity": "info",
                    "message": f"AOV in {mkt} is {((vs['aov']/avg_aov-1)*100):.0f}% above the average. Double down on bundles and upsells.",
                    "market": mkt,
                })
            if vs.get("gross_margin_pct", 0) < GM_TARGET:
                shopify_obs.append({
                    "type": "low_margin",
                    "severity": "warning",
                    "message": f"Gross margin in {mkt} is {vs['gross_margin_pct']:.1f}% (< {GM_TARGET}%). Renegotiate COGS or adjust pricing.",
                    "market": mkt,
                })
            if vs.get("fee_pct", 0) > FEES_TARGET:
                shopify_obs.append({
                    "type": "high_processing_fees",
                    "severity": "warning",
                    "message": f"Processing fees in {mkt} are {vs['fee_pct']:.1f}% (> {FEES_TARGET}%). Optimize gateway rates.",
                    "market": mkt,
                })

    # Facebook observations (spend share, CAC, MER)
    facebook_obs: List[Dict[str, Any]] = []
    for mkt, vs in per_market.items():
        meta_spend = vs.get("meta_spend", 0.0)
        total_spend = vs.get("ad_spend", 0.0)
        orders = vs.get("orders", 0) or 0
        cpa_meta = (meta_spend / orders) if orders else 0.0
        mer = vs.get("mer", 0.0)
        spend_share = (meta_spend / total_spend * 100.0) if total_spend else 0.0
        if meta_spend > 0:
            msg = f"{mkt}: Meta spend ${meta_spend:,.0f} ({spend_share:.0f}% of spend), CAC ${cpa_meta:.0f}, MER {mer:.2f}."
            sev = "info"
            if cpa_meta > CPA_META_TARGET:
                sev = "warning"
                msg += " CAC above target — iterate creatives/audiences."
            elif mer >= 3:
                sev = "success"
                msg += " Strong MER — consider scaling budget."
            facebook_obs.append({"type": "meta_market", "severity": sev, "message": msg, "market": mkt})

    # Consolidated top-level observations
    consolidated: List[Dict[str, Any]] = []
    if current.get("aov", 0) >= AOV_TARGET:
        consolidated.append({"type": "aov_ok", "severity": "success", "message": f"AOV is healthy at ${current['aov']:.0f}."})
    else:
        consolidated.append({"type": "aov_low", "severity": "warning", "message": f"AOV below target (${current['aov']:.0f} < ${AOV_TARGET:.0f}). Test bundles, tiered pricing, and post-purchase upsells."})

    if current.get("gross_margin_pct", 0) >= GM_TARGET:
        consolidated.append({"type": "margin_ok", "severity": "success", "message": f"Gross margin {current['gross_margin_pct']:.0f}% meets target."})
    else:
        consolidated.append({"type": "margin_low", "severity": "warning", "message": f"Gross margin {current['gross_margin_pct']:.0f}% below {GM_TARGET}%. Review COGS, discounts, and pricing."})

    if current.get("fee_pct", 0) <= FEES_TARGET:
        consolidated.append({"type": "fees_ok", "severity": "success", "message": f"Processing fees {current['fee_pct']:.1f}% within guardrail."})
    else:
        consolidated.append({"type": "fees_high", "severity": "warning", "message": f"Processing fees {current['fee_pct']:.1f}% exceed {FEES_TARGET}%. Consider better rates or mix."})

    # Return payload friendly for n8n
    payload = {
        "range": {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")},
        "market": market,
        "observations": consolidated,
        "facebook": facebook_obs,
        "shopify": shopify_obs,
        "per_market": {
            mkt: {
                "revenue": round(v.get("revenue", 0.0), 2),
                "orders": int(v.get("orders", 0)),
                "ad_spend": round(v.get("ad_spend", 0.0), 2),
                "meta_spend": round(v.get("meta_spend", 0.0), 2),
                "profit": round(v.get("profit", 0.0), 2),
                "aov": round(v.get("aov", 0.0), 2),
                "gross_margin_pct": round(v.get("gross_margin_pct", 2), 2),
                "fee_pct": round(v.get("fee_pct", 2), 2),
                "mer": round(v.get("mer", 0.0), 2),
                "cpa": round(v.get("cpa", 0.0), 2),
                "cpa_meta": round(v.get("cpa_meta", 0.0), 2),
                "channel_spend": v.get("channel_spend", {}),
            }
            for mkt, v in per_market.items()
        },
    }
    return payload


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
