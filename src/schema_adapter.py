"""Universal inventory schema adapter.

Maps arbitrary CSV/XLSX column names into the legacy canonical structure used
by the existing SmartInventory pipeline. Existing downstream modules remain
unchanged. Mapping is heuristic-first with an optional Gemini enhancement.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


ALIASES = {
    "Product_ID": ["product id", "product code", "item id", "item code", "sku id", "barcode", "material code"],
    "SKU": ["sku", "stock keeping unit", "item sku", "product sku"],
    "Product_Name": ["product name", "item name", "description", "material name", "product"],
    "Category": ["category", "product category", "item category", "department", "segment"],
    "Subcategory": ["subcategory", "sub category", "sub-category"],
    "Brand": ["brand", "manufacturer", "make"],
    "Supplier": ["supplier", "vendor", "supplier name", "vendor name"],
    "Store_ID": ["store id", "warehouse id", "location id", "branch id"],
    "Store_City": ["store city", "city", "location", "warehouse", "branch"],
    "Batch_No": ["batch no", "batch number", "lot", "lot number", "batch id"],
    "Purchase_Price": ["purchase price", "cost price", "unit cost", "cost", "buying price", "cogs"],
    "Selling_Price": ["selling price", "sale price", "unit price", "retail price", "mrp", "price"],
    "Current_Stock": ["current stock", "stock quantity", "quantity on hand", "on hand", "inventory", "stock", "qty"],
    "Reorder_Level": ["reorder level", "reorder point", "minimum stock", "min stock", "safety stock"],
    "Daily_Sales": ["daily sales", "sales velocity", "avg daily sales", "average daily demand", "units sold per day"],
    "Weekly_Demand": ["weekly demand", "weekly sales"],
    "Monthly_Demand": ["monthly demand", "monthly sales", "past sales volume", "units sold"],
    "Lead_Time_Days": ["lead time days", "lead time", "supplier lead time"],
    "Manufacturing_Date": ["manufacturing date", "mfg date", "production date", "manufactured on"],
    "Expiry_Date": ["expiry date", "expiration date", "best before", "use by", "shelf life date"],
    "Timestamp": ["timestamp", "date added", "transaction date", "record date", "date"],
    "Discount_%": ["discount", "discount percent", "discount percentage", "markdown"],
    "Customer_Rating": ["customer rating", "rating", "review score"],
}

CORE = {"Product_ID", "Current_Stock", "Purchase_Price", "Selling_Price"}
RECOMMENDED = CORE | {"Product_Name", "Category", "Daily_Sales", "Reorder_Level", "Expiry_Date"}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


@dataclass
class SchemaReport:
    mappings: dict[str, str] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    derived_fields: list[str] = field(default_factory=list)
    defaulted_fields: list[str] = field(default_factory=list)
    unmapped_source_columns: list[str] = field(default_factory=list)
    compatibility_score: float = 0.0
    supported_analyses: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _heuristic_map(columns: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    mapping, methods, used = {}, {}, set()
    normalized = {c: _norm(c) for c in columns}
    for canonical, aliases in ALIASES.items():
        candidates = [_norm(canonical), *map(_norm, aliases)]
        exact = next((c for c, n in normalized.items() if c not in used and n in candidates), None)
        if exact:
            mapping[canonical], methods[canonical] = exact, "alias"
            used.add(exact)
            continue
        best_col, best_score = None, 0.0
        for col, norm_col in normalized.items():
            if col in used:
                continue
            score = max(SequenceMatcher(None, norm_col, a).ratio() for a in candidates)
            if score > best_score:
                best_col, best_score = col, score
        if best_col and best_score >= 0.78:
            mapping[canonical], methods[canonical] = best_col, f"fuzzy:{best_score:.2f}"
            used.add(best_col)
    return mapping, methods


def _llm_map(columns: list[str], existing: dict[str, str]) -> dict[str, str]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return {}
    try:
        from google import genai
        client = genai.Client(api_key=key)
        prompt = (
            "Map unfamiliar inventory spreadsheet columns to canonical fields. "
            "Return only JSON {\"Canonical_Field\":\"source column\"}. Never map a source twice. "
            f"Canonical fields: {list(ALIASES)}. Source columns: {columns}. Already mapped: {existing}."
        )
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = (response.text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        used = set(existing.values())
        return {k: v for k, v in parsed.items() if k in ALIASES and v in columns and v not in used}
    except Exception:
        return {}


def standardize_dataset(source: pd.DataFrame) -> tuple[pd.DataFrame, SchemaReport]:
    raw = source.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    mapping, methods = _heuristic_map(raw.columns.tolist())
    for canonical, source_col in _llm_map(raw.columns.tolist(), mapping).items():
        if canonical not in mapping:
            mapping[canonical], methods[canonical] = source_col, "gemini"

    out = raw.copy()
    for canonical, source_col in mapping.items():
        if canonical not in out.columns:
            out[canonical] = out[source_col]

    report = SchemaReport(mappings=mapping, methods=methods)
    today = pd.Timestamp.today().normalize()

    def default(name, value):
        if name not in out.columns:
            out[name] = value
            report.defaulted_fields.append(name)

    # Identity/descriptive defaults are safe and explicitly reported.
    default("Product_ID", [f"ROW-{i+1:06d}" for i in range(len(out))])
    default("SKU", out["Product_ID"].astype(str))
    default("Product_Name", out["Product_ID"].astype(str))
    default("Category", "General")
    default("Subcategory", "General")
    default("Brand", "Unknown")
    default("Supplier", "Unknown")
    default("Store_ID", "MAIN")
    default("Store_City", "Unknown")
    default("Batch_No", out["Product_ID"].astype(str))

    numeric_defaults = {
        "Purchase_Price": 0.0, "Selling_Price": 0.0, "Current_Stock": 0.0,
        "Reorder_Level": 0.0, "Daily_Sales": 0.0, "Weekly_Demand": 0.0,
        "Monthly_Demand": 0.0, "Lead_Time_Days": 7, "Discount_%": 0.0,
        "Customer_Rating": 0.0, "Revenue_At_Risk": 0.0, "Recovered_Revenue": 0.0,
        "Waste_Cost": 0.0, "Demand_Forecast": 0.0,
    }
    for col, value in numeric_defaults.items():
        default(col, value)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)

    # Business-valid derivations when a stronger source exists.
    if "Daily_Sales" in report.defaulted_fields and "Monthly_Demand" not in report.defaulted_fields:
        out["Daily_Sales"] = out["Monthly_Demand"] / 30
        report.derived_fields.append("Daily_Sales")
        report.defaulted_fields.remove("Daily_Sales")
    if "Weekly_Demand" in report.defaulted_fields and "Daily_Sales" not in report.defaulted_fields:
        out["Weekly_Demand"] = out["Daily_Sales"] * 7
        report.derived_fields.append("Weekly_Demand")
        report.defaulted_fields.remove("Weekly_Demand")
    if "Monthly_Demand" in report.defaulted_fields and "Daily_Sales" not in report.defaulted_fields:
        out["Monthly_Demand"] = out["Daily_Sales"] * 30
        report.derived_fields.append("Monthly_Demand")
        report.defaulted_fields.remove("Monthly_Demand")
    if "Reorder_Level" in report.defaulted_fields and "Daily_Sales" not in report.defaulted_fields:
        out["Reorder_Level"] = np.ceil(out["Daily_Sales"] * out["Lead_Time_Days"] * 1.5)
        report.derived_fields.append("Reorder_Level")
        report.defaulted_fields.remove("Reorder_Level")

    default("Timestamp", today)
    default("Manufacturing_Date", today - pd.Timedelta(days=365))
    default("Expiry_Date", today + pd.Timedelta(days=365))
    for col in ["Timestamp", "Manufacturing_Date", "Expiry_Date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").fillna(today)
    out["Days_Remaining"] = (out["Expiry_Date"] - today).dt.days
    out["Expiry_Risk"] = np.select(
        [out["Days_Remaining"] < 30, out["Days_Remaining"] <= 90],
        ["High", "Medium"], default="Low"
    )
    out["Overstock_Level"] = np.maximum(out["Reorder_Level"] * 3, out["Current_Stock"])
    out["Recommended_Action"] = "Analyze"
    for col in ["Promotion_Type", "Season", "Holiday", "Weather", "Prediction_Result"]:
        default(col, "Unknown")

    mapped_recommended = len(RECOMMENDED & set(mapping))
    report.compatibility_score = round(mapped_recommended / len(RECOMMENDED) * 100, 1)
    report.unmapped_source_columns = [c for c in raw.columns if c not in mapping.values()]
    if {"Current_Stock", "Purchase_Price"} <= set(mapping): report.supported_analyses.append("inventory_value")
    if {"Current_Stock", "Selling_Price", "Expiry_Date"} <= set(mapping): report.supported_analyses.append("expiry_revenue_risk")
    if {"Current_Stock", "Daily_Sales"} <= set(mapping): report.supported_analyses.append("demand_and_stock")
    if {"Purchase_Price", "Selling_Price"} <= set(mapping): report.supported_analyses.append("profitability")
    missing_core = CORE - set(mapping)
    if missing_core:
        report.warnings.append("Core fields not mapped: " + ", ".join(sorted(missing_core)) + ". Related outputs use explicit defaults and must not be treated as financial truth.")
    out["_Schema_Compatibility_Pct"] = report.compatibility_score
    return out, report
