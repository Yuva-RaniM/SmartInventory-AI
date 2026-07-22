import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_raw_dataset(output_path, num_records=5000):
    print(f"Generating {num_records} synthetic raw records...")
    np.random.seed(42)
    
    # Categories and corresponding brands/products
    cat_brand_prod = {
        "Electronics": {
            "Apple": ["iPhone 13", "iPhone 14 Pro", "MacBook Air", "iPad Pro", "Apple Watch"],
            "Samsung": ["Galaxy S22", "Galaxy S23 Ultra", "Galaxy Tab S8", "Galaxy Watch 5"],
            "Dell": ["XPS 15", "Inspiron 14", "Latitude 5430"],
            "Sony": ["WH-1000XM4 Headphones", "PlayStation 5", "Bravia TV"]
        },
        "Groceries": {
            "Nestle": ["Whole Milk", "Greek Yogurt", "Chocolate Bar", "Instant Coffee"],
            "Kraft": ["Mac & Cheese", "Mayonnaise", "Cheddar Cheese"],
            "Dole": ["Organic Bananas", "Canned Pineapples", "Fresh Strawberries"],
            "General Mills": ["Cheerios Cereal", "Oatmeal Packets", "Fiber One Bars"]
        },
        "Pharmaceuticals": {
            "Pfizer": ["Advil Ibuprofen", "Multivitamin Tablets", "Cough Syrup"],
            "Bayer": ["Aspirin", "Claritin Allergy", "Aleve Pain Reliever"],
            "Johnson & Johnson": ["Band-Aid Strips", "Baby Shampoo", "Neutrogena Cleanser"]
        },
        "Clothing": {
            "Nike": ["Air Max Sneakers", "Dry-Fit T-Shirt", "Running Shorts", "Athletic Socks"],
            "Adidas": ["Ultraboost Shoes", "Trefoil Hoodie", "Track Pants"],
            "Levi's": ["501 Original Jeans", "Denim Jacket", "Graphic Tee"]
        },
        "Home & Kitchen": {
            "Keurig": ["Coffee Maker", "K-Cup Pods"],
            "Dyson": ["V12 Vacuum Cleaner", "Pure Cool Fan"],
            "Instant Pot": ["Multi-Cooker", "Air Fryer Duo"]
        }
    }
    
    suppliers = ["TechDistributors Inc", "Global Trade Corp", "Apex Logistics", "Nike Retail Co", "Super Goods Ltd", "BioPharma Suppliers", "AgroFoods Co"]
    store_locations = ["Warehouse A", "Warehouse B", "Store Front East", "Store Front West", "Central Depot"]
    
    data = []
    
    # Date generation
    today = datetime.now()
    
    for i in range(num_records):
        # Pick category
        category = np.random.choice(list(cat_brand_prod.keys()))
        # Pick brand
        brand = np.random.choice(list(cat_brand_prod[category].keys()))
        # Pick product
        product_name = np.random.choice(cat_brand_prod[category][brand])
        
        # Introduce case inconsistency and whitespace anomalies in product names & categories
        name_anomaly = np.random.rand()
        if name_anomaly < 0.15:
            p_name = product_name.lower() + "  "
        elif name_anomaly < 0.25:
            p_name = "  " + product_name.upper()
        else:
            p_name = product_name
            
        cat_anomaly = np.random.rand()
        if cat_anomaly < 0.15:
            p_cat = category.lower()
        elif cat_anomaly < 0.25:
            p_cat = category.upper()
        else:
            p_cat = category
            
        # Product ID
        product_id = f"PRD{10000 + i}"
        
        # Cost Price and Selling Price
        cost_price = round(np.random.uniform(5.0, 800.0), 2)
        # 5% chance of invalid cost price (negative or zero)
        if np.random.rand() < 0.05:
            cost_price = np.random.choice([-10.0, 0.0, -50.0])
            
        # Selling price is cost price + markup (typically 20% to 50%)
        # 5% chance of selling price being lower than cost price (invalid rule)
        if np.random.rand() < 0.05:
            selling_price = round(cost_price * 0.8, 2)
        else:
            selling_price = round(cost_price * np.random.uniform(1.2, 1.6), 2)
            
        # 5% chance of missing selling price (NaN)
        if np.random.rand() < 0.05:
            selling_price = np.nan
            
        # Stock quantities
        stock_qty = int(np.random.randint(0, 1500))
        # 3% chance of negative stock quantity
        if np.random.rand() < 0.03:
            stock_qty = np.random.choice([-5, -20, -100])
        # 3% chance of missing stock quantity
        if np.random.rand() < 0.03:
            stock_qty = np.nan
            
        # Min and Max levels
        min_stock = int(np.random.randint(20, 150))
        max_stock = int(min_stock * np.random.uniform(4.0, 8.0))
        
        # Sales Velocity (units sold per day)
        sales_velocity = round(np.random.exponential(scale=5.0) + 0.1, 2)
        
        # Past sales volume (to compute historical Revenue and Profit)
        past_sales_vol = int(np.random.randint(10, 500))
        
        # Date Added (last 2 years)
        days_ago = np.random.randint(1, 730)
        date_added_dt = today - timedelta(days=days_ago)
        
        # Format Date Added randomly to test parser resilience
        date_fmt = np.random.choice(["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"])
        date_added = date_added_dt.strftime(date_fmt)
        
        # Expiry Date (only for Groceries & Pharmaceuticals)
        expiry_date = ""
        if category in ["Groceries", "Pharmaceuticals"]:
            # Some expired, some expiring soon (0-60 days), some far (60-365 days)
            expiry_type = np.random.choice(["expired", "near", "safe"], p=[0.1, 0.25, 0.65])
            if expiry_type == "expired":
                expiry_dt = today - timedelta(days=np.random.randint(1, 90))
            elif expiry_type == "near":
                expiry_dt = today + timedelta(days=np.random.randint(1, 60))
            else:
                expiry_dt = today + timedelta(days=np.random.randint(61, 365))
            
            # Format expiry date
            expiry_date = expiry_dt.strftime(np.random.choice(["%Y-%m-%d", "%d/%m/%Y"]))
        
        # Last Restock Date (must be >= Date Added)
        restock_dt = date_added_dt + timedelta(days=np.random.randint(0, max(1, days_ago)))
        last_restock = restock_dt.strftime("%Y-%m-%d")
        
        # Supplier
        supplier = np.random.choice(suppliers)
        
        # Location
        location = np.random.choice(store_locations)
        
        # Status
        status = np.random.choice(["Active", "Discontinued", "active", "discontinued"], p=[0.85, 0.1, 0.03, 0.02])
        
        data.append([
            product_id, p_name, p_cat, brand, supplier, cost_price, 
            selling_price, stock_qty, min_stock, max_stock, sales_velocity,
            past_sales_vol, date_added, expiry_date, last_restock, location, status
        ])
        
    # Create DataFrame
    columns = [
        "Product_ID", "Product_Name", "Category", "Brand", "Supplier", "Cost_Price",
        "Selling_Price", "Stock_Quantity", "Min_Stock_Level", "Max_Stock_Level", "Sales_Velocity",
        "Past_Sales_Volume", "Date_Added", "Expiry_Date", "Last_Restock_Date", "Store_Location", "Status"
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Introduce duplicate rows (around 3% duplicates)
    dup_indices = np.random.choice(df.index, size=int(num_records * 0.03), replace=False)
    df_dups = df.iloc[dup_indices].copy()
    # Modify IDs slightly to make some near-duplicates, but also add exact duplicates
    exact_dups = df_dups.sample(frac=0.5, random_state=42)
    df = pd.concat([df, exact_dups], ignore_index=True)
    
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Ensure folder structure exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved successfully to {output_path} (Shape: {df.shape})")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    output_file = os.path.join(output_dir, "SmartInventory_AI_Raw_Dataset.csv")
    generate_raw_dataset(output_file, num_records=50000)
