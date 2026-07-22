import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, MetaData, Table
from sqlalchemy.orm import declarative_base, sessionmaker

# Define SQLAlchemy Base
Base = declarative_base()

# 1. Inventory Table Schema
class InventoryModel(Base):
    __tablename__ = "inventory"
    
    Product_ID = Column(String(50), primary_key=True)
    Product_Name = Column(String(100), nullable=False)
    Category = Column(String(50))
    Brand = Column(String(50))
    Supplier = Column(String(100))
    Cost_Price = Column(Float)
    Selling_Price = Column(Float)
    Stock_Quantity = Column(Integer)
    Min_Stock_Level = Column(Integer)
    Max_Stock_Level = Column(Integer)
    Sales_Velocity = Column(Float)
    Past_Sales_Volume = Column(Integer)
    Date_Added = Column(DateTime)
    Expiry_Date = Column(DateTime, nullable=True)
    Last_Restock_Date = Column(DateTime)
    Store_Location = Column(String(50))
    Status = Column(String(50))
    Inventory_Value = Column(Float)
    Revenue = Column(Float)
    Profit = Column(Float)
    Profit_Margin = Column(Float)
    Demand_Level = Column(String(20))
    Stock_Status = Column(String(20))
    Expiry_Risk = Column(String(20))
    Revenue_at_Risk = Column(Float)
    Recoverable_Revenue = Column(Float)
    Inventory_Turnover = Column(Float)
    Low_Stock_Flag = Column(Integer)
    Overstock_Flag = Column(Integer)
    Reorder_Flag = Column(Integer)
    ABC_Category = Column(String(5))
    Inventory_Health = Column(Integer)

# 2. Prediction Table Schema
class PredictionModel(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    Product_ID = Column(String(50), ForeignKey("inventory.Product_ID"))
    Predicted_Expiry_Risk = Column(String(20))
    Predicted_Sales_Velocity = Column(Float)
    Predicted_Revenue = Column(Float)
    Predicted_Cluster = Column(Integer)
    Predicted_Revenue_at_Risk = Column(Float)
    Predicted_Inventory_Health = Column(Integer)
    Prediction_Date = Column(DateTime, default=datetime.now)

# 3. Recommendation Table Schema
class RecommendationModel(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    Product_ID = Column(String(50), ForeignKey("inventory.Product_ID"))
    Recommendation_Type = Column(String(50))
    Action = Column(String(250))
    Expected_Revenue_Recovery = Column(Float)
    Business_Reason = Column(String(1000))
    Priority_Level = Column(String(20))
    Created_At = Column(DateTime, default=datetime.now)

# 4. Revenue Table Schema
class RevenueModel(Base):
    __tablename__ = "revenue"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    Product_ID = Column(String(50), ForeignKey("inventory.Product_ID"))
    Historical_Revenue = Column(Float)
    Predicted_Revenue = Column(Float)
    Report_Date = Column(DateTime, default=datetime.now)

def get_db_engine():
    """
    Creates db engine.
    Checks environment for MYSQL_DATABASE_URL.
    Defaults to SQLite for local development (final-year demonstration).
    """
    mysql_url = os.environ.get("MYSQL_DATABASE_URL")
    if mysql_url:
        print(f"Connecting to MySQL database using provided URL...")
        return create_engine(mysql_url)
    else:
        # Resolve path to project directory
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(src_dir)
        db_path = os.path.join(project_dir, "smart_inventory.db")
        print(f"Connecting to SQLite database at: {db_path} (Local Dev Fallback)")
        return create_engine(f"sqlite:///{db_path}")

def init_db(engine):
    """Initializes the schema structures in the database."""
    Base.metadata.create_all(engine)
    print("Database tables initialized successfully!")

def load_data_to_db(engine, engineered_csv, prediction_csv, recommendation_csv):
    """Loads CSV files into corresponding relational database tables."""
    print("Loading datasets into database...")
    
    # 1. Populate Inventory
    if os.path.exists(engineered_csv):
        df_inv = pd.read_csv(engineered_csv)
        df_inv["Date_Added"] = pd.to_datetime(df_inv["Date_Added"])
        df_inv["Expiry_Date"] = pd.to_datetime(df_inv["Expiry_Date"])
        df_inv["Last_Restock_Date"] = pd.to_datetime(df_inv["Last_Restock_Date"])
        
        # Write to SQL
        df_inv.to_sql("inventory", engine, if_exists="replace", index=False)
        print(f"  Inserted {len(df_inv)} records into 'inventory' table.")
    else:
        print(f"  [Error] Engineered CSV not found at {engineered_csv}")
        
    # 2. Populate Predictions
    if os.path.exists(prediction_csv):
        df_pred = pd.read_csv(prediction_csv)
        
        # Prepare df for sql insert matches schema
        df_pred_db = pd.DataFrame({
            "Product_ID": df_pred["Product_ID"],
            "Predicted_Expiry_Risk": df_pred["Predicted_Expiry_Risk"],
            "Predicted_Sales_Velocity": df_pred["Predicted_Sales_Velocity"],
            "Predicted_Revenue": df_pred["Predicted_Revenue"],
            "Predicted_Cluster": df_pred["Predicted_Cluster"],
            "Predicted_Revenue_at_Risk": df_pred["Predicted_Revenue_at_Risk"],
            "Predicted_Inventory_Health": df_pred["Predicted_Inventory_Health"],
            "Prediction_Date": datetime.now()
        })
        
        df_pred_db.to_sql("predictions", engine, if_exists="replace", index=False)
        print(f"  Inserted {len(df_pred_db)} records into 'predictions' table.")
        
        # 4. Populate Revenue
        df_rev_db = pd.DataFrame({
            "Product_ID": df_pred["Product_ID"],
            "Historical_Revenue": df_pred["Revenue"],
            "Predicted_Revenue": df_pred["Predicted_Revenue"],
            "Report_Date": datetime.now()
        })
        df_rev_db.to_sql("revenue", engine, if_exists="replace", index=False)
        print(f"  Inserted {len(df_rev_db)} records into 'revenue' table.")
    else:
        print(f"  [Error] Prediction CSV not found at {prediction_csv}")
        
    # 3. Populate Recommendations
    if os.path.exists(recommendation_csv):
        df_recs = pd.read_csv(recommendation_csv)
        
        df_recs_db = pd.DataFrame({
            "Product_ID": df_recs["Product_ID"],
            "Recommendation_Type": df_recs["Recommendation_Type"],
            "Action": df_recs["Action"],
            "Expected_Revenue_Recovery": df_recs["Expected_Revenue_Recovery"],
            "Business_Reason": df_recs["Business_Reason"],
            "Priority_Level": df_recs["Priority_Level"],
            "Created_At": datetime.now()
        })
        
        df_recs_db.to_sql("recommendations", engine, if_exists="replace", index=False)
        print(f"  Inserted {len(df_recs_db)} records into 'recommendations' table.")
    else:
        print(f"  [Error] Recommendation CSV not found at {recommendation_csv}")

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    eng_csv = os.path.join(project_dir, "data", "processed", "SmartInventory_AI_Feature_Engineered.csv")
    pred_csv = os.path.join(project_dir, "data", "processed", "Prediction.csv")
    recs_csv = os.path.join(project_dir, "data", "processed", "Recommendation_Report.csv")
    
    engine = get_db_engine()
    init_db(engine)
    load_data_to_db(engine, eng_csv, pred_csv, recs_csv)
