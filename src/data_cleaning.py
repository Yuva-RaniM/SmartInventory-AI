
import os
import pandas as pd


# --------------------------------------------------
# Remove Duplicate Records
# --------------------------------------------------
def remove_duplicates(df):
    """Removes duplicate rows."""

    initial_rows = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    removed = initial_rows - len(df)

    print(f"[Duplicates] Removed {removed} duplicate rows.")

    return df


# --------------------------------------------------
# Handle Missing Values
# --------------------------------------------------
def handle_missing_values(df):
    """Handle missing values."""

    df = df.copy()

    # Remove rows without Product ID or Product Name
    df = df.dropna(subset=["Product_ID", "Product_Name"])

    # Fill categorical values
    fill_values = {
        "Category": "Other",
        "Subcategory": "Other",
        "Brand": "Generic",
        "Supplier": "Unknown",
        "Store_City": "Unknown",
        "Promotion_Type": "None",
        "Holiday": "No",
        "Weather": "Unknown",
        "Prediction_Result": "Unknown",
        "Recommended_Action": "Normal",
        "Expiry_Risk": "Low",
        "Overstock_Level": "Low"
    }

    df.fillna(fill_values, inplace=True)

    # Remove rows if both prices are missing
    df = df.dropna(subset=["Purchase_Price", "Selling_Price"], how="all")

    # Estimate missing purchase price
    df["Purchase_Price"] = df["Purchase_Price"].fillna(
        df["Selling_Price"] / 1.35
    )

    # Estimate missing selling price
    df["Selling_Price"] = df["Selling_Price"].fillna(
        df["Purchase_Price"] * 1.35
    )

    numeric_cols = [
        "Current_Stock",
        "Reorder_Level",
        "Daily_Sales",
        "Weekly_Demand",
        "Monthly_Demand",
        "Demand_Forecast",
        "Revenue_At_Risk",
        "Waste_Cost",
        "Recovered_Revenue",
        "Discount_%",
        "Customer_Rating"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print("[Missing Values] Completed.")

    return df


# --------------------------------------------------
# Remove Invalid Records
# --------------------------------------------------
def remove_invalid_records(df):
    """Remove invalid rows."""

    df = df.copy()

    initial_rows = len(df)

    df = df[df["Purchase_Price"] > 0]
    df = df[df["Selling_Price"] > 0]

    df = df[df["Current_Stock"] >= 0]
    df = df[df["Reorder_Level"] >= 0]

    df = df[df["Daily_Sales"] >= 0]
    df = df[df["Weekly_Demand"] >= 0]
    df = df[df["Monthly_Demand"] >= 0]

    df = df[df["Lead_Time_Days"] >= 0]

    removed = initial_rows - len(df)

    print(f"[Invalid Records] Removed {removed} rows.")

    return df


# --------------------------------------------------
# Standardize Text Columns
# --------------------------------------------------
def standardize_strings(df):
    """Standardize string columns."""

    df = df.copy()

    columns = [
        "Product_Name",
        "Category",
        "Subcategory",
        "Brand",
        "Supplier",
        "Store_City",
        "Overstock_Level",
        "Expiry_Risk",
        "Recommended_Action",
        "Promotion_Type",
        "Season",
        "Holiday",
        "Weather",
        "Prediction_Result"
    ]

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
            )

    print("[String Standardization] Completed.")

    return df


# --------------------------------------------------
# Convert Dates
# --------------------------------------------------
def convert_date_columns(df):
    """Convert date columns."""

    df = df.copy()

    df["Manufacturing_Date"] = pd.to_datetime(
        df["Manufacturing_Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Expiry_Date"] = pd.to_datetime(
        df["Expiry_Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    initial_rows = len(df)

    df = df.dropna(subset=[
        "Manufacturing_Date",
        "Expiry_Date"
    ])

    removed = initial_rows - len(df)

    print(f"[Date Conversion] Removed {removed} invalid rows.")

    return df


# --------------------------------------------------
# Business Rule Validation
# --------------------------------------------------
def validate_business_rules(df):
    """Validate inventory business rules."""

    df = df.copy()

    initial_rows = len(df)

    # Selling price should not be less than purchase price
    df = df[df["Selling_Price"] >= df["Purchase_Price"]]

    # Manufacturing date must be before expiry date
    df = df[df["Manufacturing_Date"] <= df["Expiry_Date"]]

    # Days remaining cannot be negative
    df = df[df["Days_Remaining"] >= 0]

    # Customer rating between 0 and 5
    df = df[
        (df["Customer_Rating"] >= 0) &
        (df["Customer_Rating"] <= 5)
    ]

    # Discount between 0 and 100
    df = df[
        (df["Discount_%"] >= 0) &
        (df["Discount_%"] <= 100)
    ]

    removed = initial_rows - len(df)

    print(f"[Business Rules] Removed {removed} invalid rows.")

    return df


# --------------------------------------------------
# Convert Data Types
# --------------------------------------------------
def convert_data_types(df):
    """Convert columns to correct data types."""

    df = df.copy()

    string_columns = [
        "Product_ID",
        "SKU",
        "Product_Name",
        "Category",
        "Subcategory",
        "Brand",
        "Supplier",
        "Store_ID",
        "Store_City",
        "Batch_No",
        "Overstock_Level",
        "Expiry_Risk",
        "Recommended_Action",
        "Promotion_Type",
        "Season",
        "Holiday",
        "Weather",
        "Prediction_Result"
    ]

    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    integer_columns = [
        "Days_Remaining",
        "Current_Stock",
        "Reorder_Level",
        "Daily_Sales",
        "Weekly_Demand",
        "Monthly_Demand",
        "Lead_Time_Days",
        "Demand_Forecast"
    ]

    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    float_columns = [
        "Purchase_Price",
        "Selling_Price",
        "Revenue_At_Risk",
        "Discount_%",
        "Customer_Rating",
        "Waste_Cost",
        "Recovered_Revenue"
    ]

    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    print("[Data Types] Converted successfully.")

    return df


# --------------------------------------------------
# Main Cleaning Function
# --------------------------------------------------
def clean_dataset(input_path, output_path):

    print("=" * 60)
    print("SMART INVENTORY DATA CLEANING PIPELINE")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    print(f"Loaded Dataset Shape : {df.shape}")

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_invalid_records(df)
    df = standardize_strings(df)
    df = convert_date_columns(df)
    df = validate_business_rules(df)
    df = convert_data_types(df)

    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("=" * 60)
    print("Cleaning Completed Successfully")
    print(f"Final Shape : {df.shape}")
    print(f"Saved To : {output_path}")
    print("=" * 60)

    return df


# --------------------------------------------------
# Run Script
# --------------------------------------------------
if __name__ == "__main__":

    src_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(src_dir)

    raw_path = os.path.join(
        project_dir,
        "data",
        "raw",
        "SmartInventory_AI_Raw_Dataset.csv"
    )

    cleaned_path = os.path.join(
        project_dir,
        "data",
        "cleaned",
        "SmartInventory_AI_Cleaned_Dataset.csv"
    )

    clean_dataset(raw_path, cleaned_path)