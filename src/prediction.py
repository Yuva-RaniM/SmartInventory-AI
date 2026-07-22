import os
import pickle
import pandas as pd
import numpy as np

def run_predictions(input_csv_path, models_dir, output_csv_path):
    print(f"Starting Inference Pipeline on: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found at: {input_csv_path}")
        
    df = pd.read_csv(input_csv_path)
    print(f"Loaded dataset for prediction. Shape: {df.shape}")

    # Normalise column names and add legacy aliases for cleaned dataset schema
    df.columns = df.columns.str.strip()
    compatibility_columns = {
        "Cost_Price": "Purchase_Price",
        "Stock_Quantity": "Current_Stock",
        "Min_Stock_Level": "Reorder_Level",
        "Sales_Velocity": "Daily_Sales",
        "Past_Sales_Volume": "Monthly_Demand",
        "Max_Stock_Level": "Reorder_Level",
        "Days_To_Expiry": "Days_Remaining",
        "Revenue_at_Risk": "Revenue_At_Risk",
        "Recoverable_Revenue": "Recovered_Revenue"
    }
    for legacy_col, actual_col in compatibility_columns.items():
        if legacy_col not in df.columns and actual_col in df.columns:
            df[legacy_col] = df[actual_col]

    # Preprocess date columns to avoid errors if they are strings or missing
    if "Date_Added" in df.columns:
        df["Date_Added"] = pd.to_datetime(df["Date_Added"], errors="coerce")
    elif "Timestamp" in df.columns:
        df["Date_Added"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        df["Date_Added"] = pd.NaT

    if "Expiry_Date" in df.columns:
        df["Expiry_Date"] = pd.to_datetime(df["Expiry_Date"], errors="coerce")

    if "Last_Restock_Date" in df.columns:
        df["Last_Restock_Date"] = pd.to_datetime(df["Last_Restock_Date"], errors="coerce")
    else:
        df["Last_Restock_Date"] = pd.NaT

    today = pd.Timestamp.now().normalize()
    
    # Pre-calculate Days_To_Expiry (which might be used by the model features or downstream logic)
    if "Expiry_Date" in df.columns:
        df["Days_To_Expiry"] = (df["Expiry_Date"] - today).dt.days
        df["Days_To_Expiry"] = df["Days_To_Expiry"].fillna(9999).astype(int)
    else:
        df["Days_To_Expiry"] = 9999
        
    # Ensure legacy numeric aliases are available for current dataset schema
    if "Stock_Quantity" not in df.columns and "Current_Stock" in df.columns:
        df["Stock_Quantity"] = df["Current_Stock"]
    if "Cost_Price" not in df.columns and "Purchase_Price" in df.columns:
        df["Cost_Price"] = df["Purchase_Price"]
    if "Past_Sales_Volume" not in df.columns and "Monthly_Demand" in df.columns:
        df["Past_Sales_Volume"] = df["Monthly_Demand"]

    # Pre-calculate Inventory_Value (needed for revenue prediction features)
    df["Inventory_Value"] = round(df["Stock_Quantity"] * df["Cost_Price"], 2)
    df["Revenue"] = round(df["Past_Sales_Volume"] * df["Selling_Price"], 2)
    df["Profit"] = round(df["Revenue"] - (df["Past_Sales_Volume"] * df["Cost_Price"]), 2)
    df["Profit_Margin"] = np.where(df["Revenue"] > 0, round(df["Profit"] / df["Revenue"], 4), 0.0)
    
    # Pre-calculate ABC_Category (needed for demand forecast features)
    # Re-calculate ABC dynamically on the input dataset
    df_sorted = df.sort_values(by="Inventory_Value", ascending=False).copy()
    total_val = df_sorted["Inventory_Value"].sum()
    df_sorted["Cum_Value_Pct"] = df_sorted["Inventory_Value"].cumsum() / total_val if total_val > 0 else 0
    abc_map = {}
    for idx, row in df_sorted.iterrows():
        pct = row["Cum_Value_Pct"]
        if pct <= 0.80:
            abc_map[idx] = "A"
        elif pct <= 0.95:
            abc_map[idx] = "B"
        else:
            abc_map[idx] = "C"
    df["ABC_Category"] = df.index.map(abc_map).fillna("C")
    
    # Define model paths
    expiry_model_path = os.path.join(models_dir, "expiry_model.pkl")
    demand_model_path = os.path.join(models_dir, "demand_model.pkl")
    revenue_model_path = os.path.join(models_dir, "revenue_model.pkl")
    cluster_model_path = os.path.join(models_dir, "cluster_model.pkl")
    
    # Verify all model files exist
    for path in [expiry_model_path, demand_model_path, revenue_model_path, cluster_model_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Please run train_models.py first!")
            
    # Load Models
    print("Loading AI models...")
    with open(expiry_model_path, "rb") as f:
        expiry_model = pickle.load(f)
    with open(demand_model_path, "rb") as f:
        demand_model = pickle.load(f)
    with open(revenue_model_path, "rb") as f:
        revenue_model = pickle.load(f)
    with open(cluster_model_path, "rb") as f:
        cluster_model = pickle.load(f)
        
    # --- Prediction 1: Expiry Risk Category ---
    print("Predicting Expiry Risk...")
    X_exp = df[["Category", "Brand", "Cost_Price", "Selling_Price", "Stock_Quantity", "Min_Stock_Level", "Sales_Velocity"]]
    df["Predicted_Expiry_Risk"] = expiry_model.predict(X_exp)
    
    # --- Prediction 2: Demand Forecast (Predicted Sales Velocity) ---
    print("Predicting Demand (Sales Velocity)...")
    X_dem = df[["Category", "Cost_Price", "Selling_Price", "Stock_Quantity", "Min_Stock_Level", "Max_Stock_Level", "ABC_Category"]]
    df["Predicted_Sales_Velocity"] = demand_model.predict(X_dem)
    # Clip negative predictions to 0
    df["Predicted_Sales_Velocity"] = df["Predicted_Sales_Velocity"].clip(lower=0.0)
    
    # --- Prediction 3: Revenue Forecast (Predicted Future Revenue) ---
    print("Predicting Revenue...")
    # Features: Cost_Price, Selling_Price, Stock_Quantity, Sales_Velocity, Past_Sales_Volume, Inventory_Value
    # In prediction mode, we use the predicted Sales_Velocity instead of historical Sales_Velocity to project future revenue
    X_rev = df[["Cost_Price", "Selling_Price", "Stock_Quantity", "Sales_Velocity", "Past_Sales_Volume", "Inventory_Value"]].copy()
    # Replace sales velocity with predicted sales velocity to forecast
    X_rev["Sales_Velocity"] = df["Predicted_Sales_Velocity"]
    df["Predicted_Revenue"] = revenue_model.predict(X_rev)
    df["Predicted_Revenue"] = df["Predicted_Revenue"].clip(lower=0.0).round(2)
    
    # --- Prediction 4: Inventory Segmentation ---
    print("Segmenting Inventory...")
    # Features: Inventory_Value, Sales_Velocity, Past_Sales_Volume, Inventory_Turnover, Profit_Margin
    # Calculate Turnover and Margin with predicted velocity
    pred_turnover = (df["Predicted_Sales_Velocity"] * 365) / (df["Stock_Quantity"] + 1)
    
    # Past Sales profit margin (using historical sales volume)
    past_revenue = df["Past_Sales_Volume"] * df["Selling_Price"]
    past_profit = past_revenue - (df["Past_Sales_Volume"] * df["Cost_Price"])
    profit_margin = np.where(past_revenue > 0, past_profit / past_revenue, 0.0)
    
    X_clus = pd.DataFrame({
        "Inventory_Value": df["Inventory_Value"],
        "Sales_Velocity": df["Predicted_Sales_Velocity"],
        "Past_Sales_Volume": df["Past_Sales_Volume"],
        "Inventory_Turnover": pred_turnover,
        "Profit_Margin": profit_margin
    })
    
    df["Predicted_Cluster"] = cluster_model.predict(X_clus)
    
    # --- Post-Processing Derived KPI Columns ---
    # 5. Revenue at Risk: Stock_Quantity * Selling_Price if predicted expiry risk is High or Medium
    df["Predicted_Revenue_at_Risk"] = np.where(
        df["Predicted_Expiry_Risk"].isin(["High", "Medium"]),
        round(df["Stock_Quantity"] * df["Selling_Price"], 2),
        0.0
    )
    
    # 6. Inventory Health (Calculated using predicted variables)
    stock_status = np.select(
        [df["Stock_Quantity"] <= df["Min_Stock_Level"], df["Stock_Quantity"] >= df["Max_Stock_Level"]],
        ["Low Stock", "Overstock"],
        default="Optimal Stock"
    )
    
    penalty_low = np.where(stock_status == "Low Stock", -20, 0)
    penalty_over = np.where(stock_status == "Overstock", -20, 0)
    penalty_exp_h = np.where(df["Predicted_Expiry_Risk"] == "High", -40, 0)
    penalty_exp_m = np.where(df["Predicted_Expiry_Risk"] == "Medium", -20, 0)
    penalty_turnover = np.where(pred_turnover < 1.0, -10, 0)
    
    df["Predicted_Inventory_Health"] = 100 + penalty_low + penalty_over + penalty_exp_h + penalty_exp_m + penalty_turnover
    df["Predicted_Inventory_Health"] = df["Predicted_Inventory_Health"].clip(0, 100).astype(int)
    
    # Save output predictions
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions completed successfully! Saved to: {output_csv_path} (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    src_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(src_dir)
    
    cleaned_csv = os.path.join(project_dir, "data", "cleaned", "SmartInventory_AI_Cleaned_Dataset.csv")
    models_dir = os.path.join(project_dir, "models")
    prediction_output = os.path.join(project_dir, "data", "processed", "Prediction.csv")
    
    run_predictions(cleaned_csv, models_dir, prediction_output)
