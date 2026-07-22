import os
import pandas as pd
import numpy as np


def run_feature_engineering(input_path, output_path):

    df = pd.read_csv(input_path)

    numeric_columns = [
        "Purchase_Price",
        "Selling_Price",
        "Current_Stock",
        "Reorder_Level",
        "Daily_Sales",
        "Weekly_Demand",
        "Monthly_Demand",
        "Past_Sales_Volume",
        "Days_Remaining",
        "Revenue_At_Risk",
        "Recovered_Revenue",
        "Inventory_Value",
        "Revenue",
        "Profit",
        "Profit_Margin",
        "Inventory_Turnover",
        "Overstock_Level"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    today = pd.Timestamp.today().normalize()

    # Convert dates
    for col in ["Manufacturing_Date", "Expiry_Date", "Timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Days to expiry
    if "Expiry_Date" in df.columns:
        df["Days_To_Expiry"] = (df["Expiry_Date"] - today).dt.days

    # Inventory Value
    if "Current_Stock" in df.columns:
        df["Inventory_Value"] = (
            df["Current_Stock"] *
            df["Purchase_Price"]
        )

    # Revenue
    if "Daily_Sales" in df.columns:
        df["Revenue"] = (
            df["Daily_Sales"] *
            df["Selling_Price"]
        )

    # Profit
    if "Revenue" in df.columns:
        df["Profit"] = (
            df["Revenue"] -
            (df["Daily_Sales"] * df["Purchase_Price"])
        )

    # Profit Margin
    if "Profit" in df.columns:
        df["Profit_Margin"] = np.where(
            df["Revenue"] == 0,
            0,
            df["Profit"] / df["Revenue"]
        )

    # Inventory Turnover
    if "Daily_Sales" in df.columns:
        df["Inventory_Turnover"] = np.where(
            df["Current_Stock"] == 0,
            0,
            df["Daily_Sales"] * 365 / df["Current_Stock"]
        )

    # Demand Level
    df["Demand_Level"] = np.select(
        [
            df["Daily_Sales"] >= 50,
            df["Daily_Sales"] >= 20
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    # Stock Status
    df["Stock_Status"] = np.select(
        [
            df["Current_Stock"] < df["Reorder_Level"],
            df["Current_Stock"] > df["Overstock_Level"]
        ],
        [
            "Low Stock",
            "Overstock"
        ],
        default="Optimal"
    )

    # Inventory Health
    df["Inventory_Health"] = 100

    df.loc[df["Days_To_Expiry"] < 30, "Inventory_Health"] -= 30
    df.loc[df["Stock_Status"] == "Low Stock", "Inventory_Health"] -= 20
    df.loc[df["Stock_Status"] == "Overstock", "Inventory_Health"] -= 20

    # Revenue at Risk
    if "Revenue_At_Risk" in df.columns:
        df["Revenue_at_Risk"] = df["Revenue_At_Risk"]
    else:
        df["Revenue_at_Risk"] = 0

    # Recoverable Revenue
    if "Recovered_Revenue" in df.columns:
        df["Recoverable_Revenue"] = df["Recovered_Revenue"]
    else:
        df["Recoverable_Revenue"] = 0

    # ABC Analysis
    df = df.sort_values("Inventory_Value", ascending=False)

    total = df["Inventory_Value"].sum()

    df["CumPerc"] = df["Inventory_Value"].cumsum() / total

    df["ABC_Category"] = np.where(
        df["CumPerc"] <= 0.80,
        "A",
        np.where(
            df["CumPerc"] <= 0.95,
            "B",
            "C"
        )
    )

    df.drop(columns=["CumPerc"], inplace=True)

    # Preserve legacy column names for compatibility with downstream model and prediction code
    compatibility_columns = {
        "Cost_Price": "Purchase_Price",
        "Stock_Quantity": "Current_Stock",
        "Min_Stock_Level": "Reorder_Level",
        "Sales_Velocity": "Daily_Sales",
        "Days_To_Expiry": "Days_Remaining",
        "Recoverable_Revenue": "Recovered_Revenue",
        "Revenue_at_Risk": "Revenue_At_Risk",
        "Max_Stock_Level": "Reorder_Level"
    }

    for old_column, actual_column in compatibility_columns.items():
        if old_column not in df.columns and actual_column in df.columns:
            df[old_column] = df[actual_column]

    # ---------------------------------------------------------
    # MODEL COMPATIBILITY FEATURES
    # ---------------------------------------------------------
    df["Cost_Price"] = df["Purchase_Price"]
    df["Stock_Quantity"] = df["Current_Stock"]
    df["Min_Stock_Level"] = df["Reorder_Level"]
    df["Sales_Velocity"] = df["Daily_Sales"]
    df["Past_Sales_Volume"] = df["Monthly_Demand"]
    df["Days_To_Expiry"] = df["Days_Remaining"]
    df["Recoverable_Revenue"] = df["Recovered_Revenue"]
    df["Revenue_at_Risk"] = df["Revenue_At_Risk"]

    df["Inventory_Value"] = (
        df["Current_Stock"]
        * df["Purchase_Price"]
    )

    df["Revenue"] = (
        df["Daily_Sales"]
        * df["Selling_Price"]
    )

    df["Profit"] = (
        df["Daily_Sales"]
        * (
            df["Selling_Price"]
            - df["Purchase_Price"]
        )
    )

    df["Profit_Margin"] = np.where(
        df["Selling_Price"] > 0,
        ((
            df["Selling_Price"]
            - df["Purchase_Price"]
        ) / df["Selling_Price"]) * 100,
        0
    )

    df["Inventory_Turnover"] = np.where(
        df["Current_Stock"] > 0,
        (
            df["Monthly_Demand"]
            / df["Current_Stock"]
        ),
        0
    )

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    if "ABC_Category" not in df.columns:
        df["ABC_Category"] = "C"

    if "Max_Stock_Level" not in df.columns and "Reorder_Level" in df.columns:
        df["Max_Stock_Level"] = df["Reorder_Level"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Feature Engineering Completed Successfully")

    return df