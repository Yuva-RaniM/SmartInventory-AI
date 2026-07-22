import os
import json
import pandas as pd
import numpy as np

def generate_recommendations(prediction_csv_path, output_csv_path, summary_json_path):
    print(f"Starting AI Recommendation Engine on: {prediction_csv_path}")
    if not os.path.exists(prediction_csv_path):
        raise FileNotFoundError(f"Prediction dataset not found at: {prediction_csv_path}")
        
    df = pd.read_csv(prediction_csv_path)
    df.columns = df.columns.str.strip()

    zero_series = pd.Series(0, index=df.index)
    no_risk_series = pd.Series("No Risk", index=df.index)

    def get_series(column_name, fallback_series):
        if column_name in df.columns:
            return df[column_name]
        return fallback_series if isinstance(fallback_series, pd.Series) else pd.Series(fallback_series, index=df.index)

    # Compatibility aliases for cleaned dataset schema and predicted dataset schema
    column_aliases = {
        "Cost_Price": "Purchase_Price",
        "Stock_Quantity": "Current_Stock",
        "Min_Stock_Level": "Reorder_Level",
        "Max_Stock_Level": "Reorder_Level",
        "Sales_Velocity": "Daily_Sales",
        "Past_Sales_Volume": "Monthly_Demand",
        "Predicted_Sales_Velocity": "Sales_Velocity",
        "Predicted_Revenue": "Revenue",
        "Predicted_Revenue_at_Risk": "Revenue_at_Risk",
        "Predicted_Expiry_Risk": "Expiry_Risk",
        "Store_Location": "Store_City"
    }
    for expected, actual in column_aliases.items():
        if expected not in df.columns and actual in df.columns:
            df[expected] = df[actual]

    # Ensure we have required fields with safe defaults
    df["Stock_Quantity"] = get_series(
        "Stock_Quantity",
        get_series("Current_Stock", zero_series)
    ).fillna(0)

    df["Min_Stock_Level"] = get_series(
        "Min_Stock_Level",
        get_series("Reorder_Level", zero_series)
    ).fillna(0)

    df["Max_Stock_Level"] = get_series(
        "Max_Stock_Level",
        get_series("Reorder_Level", df["Min_Stock_Level"])
    ).fillna(df["Min_Stock_Level"])

    df["Cost_Price"] = get_series(
        "Cost_Price",
        get_series("Purchase_Price", zero_series)
    ).fillna(0)

    df["Selling_Price"] = get_series(
        "Selling_Price",
        zero_series
    ).fillna(0)

    df["Predicted_Expiry_Risk"] = get_series(
        "Predicted_Expiry_Risk",
        get_series("Expiry_Risk", no_risk_series)
    ).fillna("No Risk")

    df["Predicted_Sales_Velocity"] = get_series(
        "Predicted_Sales_Velocity",
        get_series("Sales_Velocity", get_series("Daily_Sales", zero_series))
    ).fillna(0)

    df["Predicted_Revenue"] = get_series(
        "Predicted_Revenue",
        get_series("Revenue", zero_series)
    ).fillna(0)

    df["Predicted_Revenue_at_Risk"] = get_series(
        "Predicted_Revenue_at_Risk",
        get_series("Revenue_at_Risk", zero_series)
    ).fillna(0)

    recs = []
    
    for idx, row in df.iterrows():
        p_id = row["Product_ID"]
        p_name = row["Product_Name"]
        cat = row["Category"]
        supplier = row.get("Supplier", "Unknown Supplier")
        location = row.get("Store_Location") or row.get("Store_City") or row.get("Store_ID", "Unknown Location")
        stock = row["Stock_Quantity"]
        min_lvl = row["Min_Stock_Level"]
        max_lvl = row["Max_Stock_Level"]
        cost = row["Cost_Price"]
        price = row["Selling_Price"]
        exp_risk = row["Predicted_Expiry_Risk"]
        pred_vel = row["Predicted_Sales_Velocity"]
        pred_rev = row["Predicted_Revenue"]
        
        # 1. Discount Recommendation (Expiry Risk)
        if exp_risk == "High":
            discount = 0.40
            recovery = round(stock * price * (1 - discount), 2)
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Discount Recommendation",
                "Action": "Apply 40% Liquidating Discount",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"Product has HIGH expiry risk (expiring in <30 days). A 40% discount will accelerate sales velocity from {pred_vel:.1f}/day to liquidate stock before absolute loss.",
                "Priority_Level": "High"
            })
        elif exp_risk == "Medium":
            discount = 0.20
            recovery = round(stock * price * (1 - discount), 2)
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Discount Recommendation",
                "Action": "Apply 20% Promotional Discount",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"Product has MEDIUM expiry risk (expiring in 31-90 days). A 20% discount is recommended to stimulate demand and protect margins.",
                "Priority_Level": "Medium"
            })
            
        # 2. Reorder Recommendation
        if stock <= min_lvl:
            reorder_qty = int(max_lvl - stock)
            recovery = round(reorder_qty * price * 0.90, 2) # Assume 90% of reordered stock sells
            priority = "High" if pred_vel >= 5.0 else "Medium"
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Reorder Recommendation",
                "Action": f"Restock {reorder_qty} Units",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"Stock level ({stock}) is below reorder point ({min_lvl}). Restocking {reorder_qty} units is critical to prevent stockouts and capture projected demand of {pred_vel:.1f}/day.",
                "Priority_Level": priority
            })
            
        # 3. Overstock Recommendation
        if stock >= max_lvl:
            excess_qty = int(stock - max_lvl)
            recovery = round(excess_qty * price * 0.75, 2) # Recover 75% of value through clearance
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Overstock Recommendation",
                "Action": "Run Overstock Clearance Campaign",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"Stock level ({stock}) exceeds warehouse capacity ({max_lvl}). Recommend clearing excess {excess_qty} units at 25% discount to free up warehouse space and capital.",
                "Priority_Level": "Medium"
            })
            
        # 4. Transfer Recommendation (if overstocked and can move to another location)
        if stock >= max_lvl:
            excess_qty = int(stock - max_lvl)
            target_loc = "Store Front East" if location != "Store Front East" else "Warehouse A"
            recovery = round(excess_qty * cost, 2) # Capital cost saved
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Transfer Recommendation",
                "Action": f"Transfer {excess_qty} Units to {target_loc}",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"Product is overstocked locally in {location}. Transferring excess {excess_qty} units to {target_loc} optimizes stock distribution and reduces storage carrying costs.",
                "Priority_Level": "Medium"
            })
            
        # 5. Supplier Recommendation
        if stock <= min_lvl and pred_vel >= 8.0:
            recovery = round(cost * stock * 0.05, 2) # 5% cost savings
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Supplier Recommendation",
                "Action": "Evaluate Lead Time Agreement / Vendor SLA",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"Product has high sales velocity ({pred_vel:.1f}/day) and is frequently low in stock. Recommend renegotiating lead-time agreements with {supplier} to build safety stock buffers.",
                "Priority_Level": "Low"
            })
            
        # 6. Purchase Planning Recommendation
        if pred_vel >= 12.0:
            recovery = round(pred_rev * 0.12, 2) # 12% margin optimization
            recs.append({
                "Product_ID": p_id,
                "Product_Name": p_name,
                "Category": cat,
                "Supplier": supplier,
                "Store_Location": location,
                "Recommendation_Type": "Purchase Planning Recommendation",
                "Action": "Establish Bulk Purchase Contract",
                "Expected_Revenue_Recovery": recovery,
                "Business_Reason": f"High demand product forecasted to generate {pred_rev:.2f} in revenue. Lock in bulk-purchase contracts with {supplier} to lower unit costs by 10-15%.",
                "Priority_Level": "High"
            })
            
    # Create DataFrame
    recs_df = pd.DataFrame(recs)
    
    # Save detailed recommendations report
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    recs_df.to_csv(output_csv_path, index=False)
    print(f"Recommendations saved to: {output_csv_path} (Count: {len(recs_df)})")
    
    # Calculate Business Impact Summary Metrics
    total_products = len(df["Product_ID"].unique())
    high_risk_products = len(df[df["Predicted_Expiry_Risk"] == "High"]) + len(df[df["Stock_Quantity"] <= df["Min_Stock_Level"]])
    # Make high risk unique
    high_risk_products = len(df[(df["Predicted_Expiry_Risk"] == "High") | (df["Stock_Quantity"] <= df["Min_Stock_Level"])]["Product_ID"].unique())
    
    rev_at_risk = df["Predicted_Revenue_at_Risk"].sum()
    # Recovery is calculated only against expiring revenue at risk. Reorder,
    # transfer and supplier opportunities are intentionally excluded because
    # adding them would double-count the same product and can exceed the entire
    # at-risk value. High-risk stock assumes a 40% markdown (60% recovery), and
    # medium-risk stock a 20% markdown (80% recovery).
    risk_value = pd.to_numeric(df["Predicted_Revenue_at_Risk"], errors="coerce").fillna(0).clip(lower=0)
    recovery_rate = np.select(
        [df["Predicted_Expiry_Risk"].eq("High"), df["Predicted_Expiry_Risk"].eq("Medium")],
        [0.60, 0.80],
        default=0.0,
    )
    expected_recovery = float((risk_value * recovery_rate).sum())
    expected_recovery = min(expected_recovery, float(rev_at_risk))

    estimated_waste_reduction = (
        (expected_recovery / rev_at_risk) * 100 if rev_at_risk > 0 else 0.0
    )
    
    overall_health = df["Predicted_Inventory_Health"].mean()
    reported_recovery = pd.to_numeric(
        get_series("Recovered_Revenue", get_series("Recoverable_Revenue", zero_series)),
        errors="coerce",
    ).fillna(0).clip(lower=0)
    # Compare recovered value only with the same current at-risk rows and cap
    # every row at its risk value. This prevents historical/cumulative source
    # fields from inflating the current-period recovery KPI.
    actual_recovered = float(np.minimum(reported_recovery, risk_value).sum())
    actual_recovered = min(actual_recovered, float(rev_at_risk))
    
    summary = {
        "Products_Analysed": int(total_products),
        "High_Risk_Products": int(high_risk_products),
        "Revenue_at_Risk": float(round(rev_at_risk, 2)),
        "Actual_Recovered_Revenue": float(round(actual_recovered, 2)),
        "Expected_Recovery": float(round(expected_recovery, 2)),
        "Estimated_Waste_Reduction_Pct": float(estimated_waste_reduction),
        "Overall_Inventory_Health_Pct": float(round(overall_health, 2)),
        "Recovery_Method": "Expiry-risk value after recommended markdown; capped at revenue at risk"
    }
    
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Business Impact Summary JSON saved to: {summary_json_path}")
    print("Summary Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    return recs_df, summary

if __name__ == "__main__":
    src_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(src_dir)
    
    pred_csv = os.path.join(project_dir, "data", "processed", "Prediction.csv")
    recs_csv = os.path.join(project_dir, "data", "processed", "Recommendation_Report.csv")
    summary_json = os.path.join(project_dir, "data", "processed", "Business_Impact_Summary.json")
    
    generate_recommendations(pred_csv, recs_csv, summary_json)
