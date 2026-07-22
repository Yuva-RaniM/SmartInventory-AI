import pandas as pd

from src.schema_adapter import standardize_dataset


def test_retail_alias_schema():
    raw = pd.DataFrame({
        "Item Code": ["A1"], "Description": ["Milk"], "Department": ["Dairy"],
        "Qty": [20], "Unit Cost": [30], "Retail Price": [42],
        "Avg Daily Sales": [4], "Best Before": ["2026-08-01"],
    })
    out, report = standardize_dataset(raw)
    assert out.loc[0, "Product_ID"] == "A1"
    assert out.loc[0, "Current_Stock"] == 20
    assert "expiry_revenue_risk" in report.supported_analyses


def test_pharmacy_alias_schema():
    raw = pd.DataFrame({
        "Material Code": ["M1"], "Material Name": ["Tablet"], "Lot Number": ["L8"],
        "On Hand": [75], "COGS": [12.5], "MRP": [25], "Use By": ["2026-10-10"],
        "Vendor Name": ["Supplier A"],
    })
    out, report = standardize_dataset(raw)
    assert out.loc[0, "Batch_No"] == "L8"
    assert out.loc[0, "Supplier"] == "Supplier A"
    assert report.compatibility_score > 50


def test_minimal_unknown_schema_stays_pipeline_safe():
    raw = pd.DataFrame({"asset": ["X"], "count": [2]})
    out, report = standardize_dataset(raw)
    required_by_legacy_pipeline = {
        "Product_ID", "Product_Name", "Category", "Brand", "Purchase_Price",
        "Selling_Price", "Current_Stock", "Reorder_Level", "Daily_Sales",
        "Monthly_Demand", "Expiry_Date", "Expiry_Risk", "Overstock_Level",
    }
    assert required_by_legacy_pipeline <= set(out.columns)
    assert report.warnings

