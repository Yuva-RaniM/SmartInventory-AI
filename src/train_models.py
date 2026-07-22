import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, silhouette_score
from xgboost import XGBRegressor

def train_and_save_models(data_path, models_dir):
    print(f"Starting Model Training Pipeline on: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature engineered dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)

    # ---------------------------------------------------------
    # PREPARE AND ENGINEER REQUIRED MODEL FEATURES
    # ---------------------------------------------------------

    df.columns = df.columns.str.strip()

    numeric_columns = [
        "Purchase_Price",
        "Selling_Price",
        "Current_Stock",
        "Reorder_Level",
        "Daily_Sales",
        "Weekly_Demand",
        "Monthly_Demand",
        "Demand_Forecast",
        "Days_Remaining",
        "Revenue_At_Risk",
        "Recovered_Revenue",
        "Waste_Cost"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    column_aliases = {
        "Cost_Price": "Purchase_Price",
        "Stock_Quantity": "Current_Stock",
        "Min_Stock_Level": "Reorder_Level",
        "Sales_Velocity": "Daily_Sales",
        "Past_Sales_Volume": "Monthly_Demand",
        "Days_To_Expiry": "Days_Remaining",
        "Recoverable_Revenue": "Recovered_Revenue",
        "Revenue_at_Risk": "Revenue_At_Risk"
    }

    for expected_column, source_column in column_aliases.items():
        if expected_column not in df.columns and source_column in df.columns:
            df[expected_column] = df[source_column]

    if "Inventory_Value" not in df.columns:
        df["Inventory_Value"] = (
            df["Current_Stock"]
            * df["Purchase_Price"]
        )

    if "Revenue" not in df.columns:
        df["Revenue"] = (
            df["Daily_Sales"]
            * df["Selling_Price"]
        )

    if "Profit" not in df.columns:
        df["Profit"] = (
            df["Daily_Sales"]
            * (
                df["Selling_Price"]
                - df["Purchase_Price"]
            )
        )

    if "Profit_Margin" not in df.columns:
        df["Profit_Margin"] = np.where(
            df["Selling_Price"] > 0,
            ((
                df["Selling_Price"]
                - df["Purchase_Price"]
            ) / df["Selling_Price"]) * 100,
            0
        )

    if "Inventory_Turnover" not in df.columns:
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

    os.makedirs(models_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Expiry Risk Prediction Model (Random Forest Classifier)
    # -------------------------------------------------------------
    print("\n--- Training Expiry Risk Model (Random Forest) ---")
    # Target: Expiry_Risk (High, Medium, Low, None)
    # Target encoding: we will predict the class name directly
    # Features: Category, Brand, Cost_Price, Selling_Price, Stock_Quantity, Min_Stock_Level, Sales_Velocity
    
    X_exp = df[[
        "Category",
        "Brand",
        "Cost_Price",
        "Selling_Price",
        "Stock_Quantity",
        "Min_Stock_Level",
        "Sales_Velocity"
    ]].copy()
    y_exp = df["Expiry_Risk"].copy()
    
    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp)
    
    preprocessor_exp = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'Brand']),
            ('num', StandardScaler(), ['Cost_Price', 'Selling_Price', 'Stock_Quantity', 'Min_Stock_Level', 'Sales_Velocity'])
        ])
        
    expiry_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_exp),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    expiry_pipeline.fit(X_train_exp, y_train_exp)
    y_pred_exp = expiry_pipeline.predict(X_test_exp)
    
    acc_exp = accuracy_score(y_test_exp, y_pred_exp)
    print(f"Expiry Model Accuracy: {acc_exp:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_exp, y_pred_exp))
    
    # Save model
    expiry_model_path = os.path.join(models_dir, "expiry_model.pkl")
    with open(expiry_model_path, "wb") as f:
        pickle.dump(expiry_pipeline, f)
    print(f"Saved expiry model to {expiry_model_path}")
    
    # -------------------------------------------------------------
    # 2. Demand Forecast Model (XGBoost Regressor)
    # -------------------------------------------------------------
    print("\n--- Training Demand Forecast Model (XGBoost) ---")
    # Target: Sales_Velocity (continuous)
    # Features: Category, Cost_Price, Selling_Price, Stock_Quantity, Min_Stock_Level, Max_Stock_Level, ABC_Category
    
    X_dem = df[["Category", "Cost_Price", "Selling_Price", "Stock_Quantity", "Min_Stock_Level", "Max_Stock_Level", "ABC_Category"]].copy()
    y_dem = df["Sales_Velocity"].copy()
    
    X_train_dem, X_test_dem, y_train_dem, y_test_dem = train_test_split(X_dem, y_dem, test_size=0.2, random_state=42)
    
    preprocessor_dem = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'ABC_Category']),
            ('num', StandardScaler(), ['Cost_Price', 'Selling_Price', 'Stock_Quantity', 'Min_Stock_Level', 'Max_Stock_Level'])
        ])
        
    demand_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_dem),
        ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42))
    ])
    
    demand_pipeline.fit(X_train_dem, y_train_dem)
    y_pred_dem = demand_pipeline.predict(X_test_dem)
    
    mse_dem = mean_squared_error(y_test_dem, y_pred_dem)
    rmse_dem = np.sqrt(mse_dem)
    r2_dem = r2_score(y_test_dem, y_pred_dem)
    
    print(f"Demand Model RMSE: {rmse_dem:.4f}")
    print(f"Demand Model R2 Score: {r2_dem:.4f}")
    
    # Save model
    demand_model_path = os.path.join(models_dir, "demand_model.pkl")
    with open(demand_model_path, "wb") as f:
        pickle.dump(demand_pipeline, f)
    print(f"Saved demand model to {demand_model_path}")
    
    # -------------------------------------------------------------
    # 3. Revenue Prediction Model (Linear Regression)
    # -------------------------------------------------------------
    print("\n--- Training Revenue Prediction Model (Linear Regression) ---")
    # Target: Revenue (continuous)
    # Features: Cost_Price, Selling_Price, Stock_Quantity, Sales_Velocity, Past_Sales_Volume, Inventory_Value
    
    revenue_features = [
        "Cost_Price",
        "Selling_Price",
        "Stock_Quantity",
        "Sales_Velocity",
        "Past_Sales_Volume",
        "Inventory_Value"
    ]

    missing_revenue_features = [
        column
        for column in revenue_features
        if column not in df.columns
    ]

    if missing_revenue_features:
        raise ValueError(
            "Missing revenue model features: "
            f"{missing_revenue_features}. "
            f"Available columns: {df.columns.tolist()}"
        )

    X_rev = df[revenue_features].copy()
    y_rev = df["Revenue"].copy()
    
    X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(X_rev, y_rev, test_size=0.2, random_state=42)
    
    preprocessor_rev = StandardScaler()
    
    revenue_pipeline = Pipeline(steps=[
        ('scaler', preprocessor_rev),
        ('regressor', LinearRegression())
    ])
    
    revenue_pipeline.fit(X_train_rev, y_train_rev)
    y_pred_rev = revenue_pipeline.predict(X_test_rev)
    
    mse_rev = mean_squared_error(y_test_rev, y_pred_rev)
    rmse_rev = np.sqrt(mse_rev)
    r2_rev = r2_score(y_test_rev, y_pred_rev)
    
    print(f"Revenue Model RMSE: {rmse_rev:.4f}")
    print(f"Revenue Model R2 Score: {r2_rev:.4f}")
    
    # Save model
    revenue_model_path = os.path.join(models_dir, "revenue_model.pkl")
    with open(revenue_model_path, "wb") as f:
        pickle.dump(revenue_pipeline, f)
    print(f"Saved revenue model to {revenue_model_path}")
    
    # -------------------------------------------------------------
    # 4. Inventory Segmentation Model (K-Means Clustering)
    # -------------------------------------------------------------
    print("\n--- Training Inventory Segmentation Model (K-Means) ---")
    # Features: Inventory_Value, Sales_Velocity, Past_Sales_Volume, Inventory_Turnover, Profit_Margin
    # Unsupervised: No train/test split needed for the standard training, but we scale and cluster

    if "Inventory_Turnover" not in df.columns:
        if "Current_Stock" in df.columns and "Sales_Velocity" in df.columns:
            df["Inventory_Turnover"] = np.where(
                df["Current_Stock"] == 0,
                0,
                df["Sales_Velocity"] * 365 / df["Current_Stock"]
            )
        elif "Stock_Quantity" in df.columns and "Sales_Velocity" in df.columns:
            df["Inventory_Turnover"] = np.where(
                df["Stock_Quantity"] == 0,
                0,
                df["Sales_Velocity"] * 365 / df["Stock_Quantity"]
            )
        else:
            df["Inventory_Turnover"] = 0

    if "Profit_Margin" not in df.columns:
        if "Revenue" in df.columns and "Profit" in df.columns:
            df["Profit_Margin"] = np.where(
                df["Revenue"] == 0,
                0,
                df["Profit"] / df["Revenue"]
            )
        elif "Revenue" in df.columns and "Daily_Sales" in df.columns and "Purchase_Price" in df.columns:
            df["Profit"] = df["Revenue"] - (df["Daily_Sales"] * df["Purchase_Price"])
            df["Profit_Margin"] = np.where(
                df["Revenue"] == 0,
                0,
                df["Profit"] / df["Revenue"]
            )
        else:
            df["Profit_Margin"] = 0

    X_clus = df[["Inventory_Value", "Sales_Velocity", "Past_Sales_Volume", "Inventory_Turnover", "Profit_Margin"]].copy()
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('kmeans', kmeans)
    ])
    
    labels = kmeans.fit_predict(X_clus)
    cluster_pipeline.fit(X_clus)
    
    sil_score = silhouette_score(cluster_pipeline.named_steps['scaler'].transform(X_clus), labels)
    print(f"K-Means Clustering Silhouette Score: {sil_score:.4f}")
    
    # Let's count elements per cluster
    df["Cluster"] = labels
    print("Cluster sizes:")
    print(df["Cluster"].value_counts().sort_index())
    
    # Save model
    cluster_model_path = os.path.join(models_dir, "cluster_model.pkl")
    with open(cluster_model_path, "wb") as f:
        pickle.dump(cluster_pipeline, f)
    print(f"Saved cluster model to {cluster_model_path}")
    
    print("\nTraining completed successfully for all 4 AI Models!")

if __name__ == "__main__":
    src_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(src_dir)
    engineered_path = os.path.join(project_dir, "data", "processed", "SmartInventory_AI_Feature_Engineered.csv")
    models_path = os.path.join(project_dir, "models")
    
    train_and_save_models(engineered_path, models_path)
