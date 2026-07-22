import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set styling for matplotlib
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.figsize": (10, 6)
})

def save_matplotlib_fig(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {filepath}")

def generate_missing_values_chart(raw_df, output_dir):
    """Plots the percentage of missing values per column in the raw dataset."""
    missing_pct = raw_df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(missing_pct) > 0:
        sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="Reds_r", ax=ax)
        ax.set_title("Missing Values Percentage by Column (Raw Dataset)")
        ax.set_xlabel("Percentage (%)")
        ax.set_ylabel("Columns")
        for i, v in enumerate(missing_pct.values):
            ax.text(v + 0.2, i, f"{v:.2f}%", va='center', fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No missing values found in raw dataset!", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax.set_title("Missing Values Analysis")
        
    save_matplotlib_fig(fig, "missing_values.png", output_dir)
    
    # Plotly version
    if len(missing_pct) > 0:
        fig_ly = px.bar(x=missing_pct.values, y=missing_pct.index, orientation='h',
                        title="Missing Values Percentage by Column",
                        labels={'x': 'Percentage (%)', 'y': 'Columns'},
                        color=missing_pct.values, color_continuous_scale='Reds')
        fig_ly.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig_ly = go.Figure()
        fig_ly.add_annotation(text="No missing values in raw dataset!", showarrow=False, font=dict(size=16))
    return fig_ly

def generate_correlation_matrix(df, output_dir):
    """Generates correlation matrix of numerical variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Drop binary flags from standard correlation to keep it clean, or keep them
    cols_to_corr = [c for c in numeric_cols if c not in ["Low_Stock_Flag", "Overstock_Flag", "Reorder_Flag"]]
    
    corr = df[cols_to_corr].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, 
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Numeric Correlation Matrix Heatmap", pad=20)
    
    save_matplotlib_fig(fig, "correlation_matrix.png", output_dir)
    
    # Plotly Version
    fig_ly = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       title="Numeric Features Correlation Heatmap", aspect="auto")
    return fig_ly

def generate_category_analysis(df, output_dir):
    """Analyzes products, value, and revenue by category."""
    cat_summary = df.groupby("Category").agg(
        Product_Count=("Product_ID", "count"),
        Inventory_Value=("Inventory_Value", "sum"),
        Revenue=("Revenue", "sum")
    ).reset_index()
    
    # Plotly multi-chart
    fig_ly = px.bar(cat_summary, x="Category", y=["Inventory_Value", "Revenue"],
                    barmode="group", title="Revenue vs Inventory Value by Category",
                    labels={"value": "Amount ($)", "variable": "Metric"})
    
    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cat_summary))
    width = 0.35
    ax.bar(x - width/2, cat_summary["Inventory_Value"], width, label="Inventory Value", color="#1f77b4")
    ax.bar(x + width/2, cat_summary["Revenue"], width, label="Revenue", color="#ff7f0e")
    ax.set_title("Revenue vs Inventory Value by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_summary["Category"], rotation=45, ha='right')
    ax.set_ylabel("Amount ($)")
    ax.legend()
    
    save_matplotlib_fig(fig, "category_analysis.png", output_dir)
    return fig_ly

def generate_brand_analysis(df, output_dir):
    """Analyzes the top 10 brands by products count and inventory value."""
    brand_summary = df.groupby("Brand").agg(
        Product_Count=("Product_ID", "count"),
        Inventory_Value=("Inventory_Value", "sum")
    ).sort_values(by="Inventory_Value", ascending=False).head(10).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=brand_summary, x="Brand", y="Inventory_Value", ax=ax1, color="#2ca02c")
    ax1.set_title("Top 10 Brands by Inventory Value & Product Count")
    ax1.set_ylabel("Inventory Value ($)")
    ax1.set_xticklabels(brand_summary["Brand"], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=brand_summary, x="Brand", y="Product_Count", ax=ax2, color="#d62728", marker="o", linewidth=2)
    ax2.set_ylabel("Product Count (Line)")
    ax2.grid(False)
    
    save_matplotlib_fig(fig, "brand_analysis.png", output_dir)
    
    # Plotly Version
    fig_ly = go.Figure()
    fig_ly.add_trace(go.Bar(x=brand_summary["Brand"], y=brand_summary["Inventory_Value"], name="Inventory Value", marker_color="#2ca02c"))
    fig_ly.add_trace(go.Scatter(x=brand_summary["Brand"], y=brand_summary["Product_Count"], name="Product Count", yaxis="y2", line=dict(color="#d62728", width=3), marker=dict(size=8)))
    fig_ly.update_layout(
        title="Top 10 Brands by Inventory Value and Product Count",
        yaxis=dict(title="Inventory Value ($)"),
        yaxis2=dict(title="Product Count", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    return fig_ly

def generate_supplier_analysis(df, output_dir):
    """Analyzes suppliers by revenue generated and inventory cost."""
    sup_summary = df.groupby("Supplier").agg(
        Revenue=("Revenue", "sum"),
        Recoverable_Revenue=("Recoverable_Revenue", "sum")
    ).sort_values(by="Revenue", ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=sup_summary, x="Revenue", y="Supplier", color="#9467bd", label="Total Revenue", ax=ax)
    sns.barplot(data=sup_summary, x="Recoverable_Revenue", y="Supplier", color="#bcbd22", label="Recoverable Revenue", ax=ax)
    ax.set_title("Supplier Revenue & Recoverable Potential")
    ax.set_xlabel("Amount ($)")
    ax.legend()
    
    save_matplotlib_fig(fig, "supplier_analysis.png", output_dir)
    
    # Plotly Version
    fig_ly = px.bar(sup_summary, y="Supplier", x=["Revenue", "Recoverable_Revenue"], 
                    orientation='h', title="Supplier Revenue Contribution & Recoverable Potential",
                    labels={"value": "Amount ($)", "variable": "Revenue Type"})
    return fig_ly

def generate_trends(df, output_dir):
    """Generates Revenue Trend, Inventory Trend, and Sales Trend over time."""
    df = df.copy()
    df["YearMonth"] = df["Date_Added"].dt.to_period("M").astype(str)
    
    # Group by month added
    trend_summary = df.groupby("YearMonth").agg(
        Revenue_Added=("Revenue", "sum"),
        Inventory_Cost_Added=("Inventory_Value", "sum"),
        Sales_Volume_Added=("Past_Sales_Volume", "sum")
    ).sort_index().reset_index()
    
    # Revenue Trend
    fig_rev, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend_summary, x="YearMonth", y="Revenue_Added", marker="o", color="#e377c2", linewidth=2.5, ax=ax)
    ax.set_title("Historical Revenue Trend (by Product Onboarding Month)")
    ax.set_xticklabels(trend_summary["YearMonth"], rotation=45)
    ax.set_ylabel("Revenue ($)")
    save_matplotlib_fig(fig_rev, "revenue_trend.png", output_dir)
    
    fig_rev_ly = px.line(trend_summary, x="YearMonth", y="Revenue_Added", markers=True,
                         title="Historical Revenue Trend (by Product Onboarding Month)",
                         labels={"Revenue_Added": "Revenue ($)", "YearMonth": "Month"})
    
    # Inventory Trend
    fig_inv, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=trend_summary, x="YearMonth", y="Inventory_Cost_Added", color="#17becf", ax=ax)
    ax.set_title("Onboarded Inventory Cost Trend")
    ax.set_xticklabels(trend_summary["YearMonth"], rotation=45)
    ax.set_ylabel("Inventory Value ($)")
    save_matplotlib_fig(fig_inv, "inventory_trend.png", output_dir)
    
    fig_inv_ly = px.bar(trend_summary, x="YearMonth", y="Inventory_Cost_Added",
                        title="Onboarded Inventory Cost Trend",
                        labels={"Inventory_Cost_Added": "Inventory Cost ($)"})
    
    # Sales Trend
    fig_sales, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend_summary, x="YearMonth", y="Sales_Volume_Added", marker="s", color="#7f7f7f", linewidth=2.5, ax=ax)
    ax.set_title("Sales Volume Trend")
    ax.set_xticklabels(trend_summary["YearMonth"], rotation=45)
    ax.set_ylabel("Units Sold")
    save_matplotlib_fig(fig_sales, "sales_trend.png", output_dir)
    
    fig_sales_ly = px.line(trend_summary, x="YearMonth", y="Sales_Volume_Added", markers=True,
                           title="Sales Volume Trend", labels={"Sales_Volume_Added": "Units Sold"})
    
    return fig_rev_ly, fig_inv_ly, fig_sales_ly

def generate_expiry_trend(df, output_dir):
    """Plots the distribution of upcoming product expirations over the next year."""
    # Filter products that expire and are not in the placeholder/9999 range
    expiring_products = df[(df["Days_To_Expiry"] < 365) & (df["Days_To_Expiry"] > -365)].copy()
    
    if len(expiring_products) == 0:
        fig_ly = go.Figure()
        fig_ly.add_annotation(text="No expiring products detected in next 365 days.", showarrow=False)
        return fig_ly
        
    expiring_products["ExpiryMonth"] = expiring_products["Expiry_Date"].dt.to_period("M").astype(str)
    
    expiry_summary = expiring_products.groupby("ExpiryMonth").agg(
        Product_Count=("Product_ID", "count"),
        Revenue_at_Risk=("Revenue_at_Risk", "sum")
    ).sort_index().reset_index()
    
    # Expiry Trend plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=expiry_summary, x="ExpiryMonth", y="Product_Count", color="#ff7f0e", ax=ax1)
    ax1.set_title("Upcoming Product Expirations & Revenue at Risk")
    ax1.set_ylabel("Number of Expiring Products")
    ax1.set_xticklabels(expiry_summary["ExpiryMonth"], rotation=45)
    
    ax2 = ax1.twinx()
    sns.lineplot(data=expiry_summary, x="ExpiryMonth", y="Revenue_at_Risk", color="#d62728", marker="d", linewidth=2.5, ax=ax2)
    ax2.set_ylabel("Revenue at Risk ($)")
    ax2.grid(False)
    
    save_matplotlib_fig(fig, "expiry_trend.png", output_dir)
    
    # Plotly Version
    fig_ly = go.Figure()
    fig_ly.add_trace(go.Bar(x=expiry_summary["ExpiryMonth"], y=expiry_summary["Product_Count"], name="Expiring Products Count", marker_color="#ff7f0e"))
    fig_ly.add_trace(go.Scatter(x=expiry_summary["ExpiryMonth"], y=expiry_summary["Revenue_at_Risk"], name="Revenue at Risk ($)", yaxis="y2", line=dict(color="#d62728", width=3), marker=dict(size=8)))
    fig_ly.update_layout(
        title="Upcoming Product Expirations & Revenue at Risk",
        yaxis=dict(title="Number of Expiring Products"),
        yaxis2=dict(title="Revenue at Risk ($)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    return fig_ly

def generate_top_rankings(df, output_dir):
    """Visualizes top selling products, categories, and suppliers."""
    # Top Selling Products
    top_sel_products = df.sort_values(by="Past_Sales_Volume", ascending=False).head(10)
    fig_p, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_sel_products, x="Past_Sales_Volume", y="Product_Name", palette="viridis", ax=ax)
    ax.set_title("Top 10 Selling Products by Volume")
    ax.set_xlabel("Units Sold")
    save_matplotlib_fig(fig_p, "top_selling_products.png", output_dir)
    
    fig_p_ly = px.bar(top_sel_products, x="Past_Sales_Volume", y="Product_Name", orientation='h',
                       title="Top 10 Selling Products by Volume",
                       labels={'Past_Sales_Volume': 'Units Sold', 'Product_Name': 'Product'})
    fig_p_ly.update_layout(yaxis={'categoryorder':'total ascending'})
    
    # Top Categories (by Revenue)
    top_categories = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(5).reset_index()
    fig_c, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_categories, x="Revenue", y="Category", palette="magma", ax=ax)
    ax.set_title("Top 5 Categories by Revenue")
    ax.set_xlabel("Total Revenue ($)")
    save_matplotlib_fig(fig_c, "top_categories.png", output_dir)
    
    fig_c_ly = px.bar(top_categories, x="Revenue", y="Category", orientation='h',
                       title="Top 5 Categories by Revenue",
                       labels={'Revenue': 'Revenue ($)', 'Category': 'Category'})
    fig_c_ly.update_layout(yaxis={'categoryorder':'total ascending'})
    
    # Top Suppliers (by Inventory Value)
    top_suppliers = df.groupby("Supplier")["Inventory_Value"].sum().sort_values(ascending=False).head(5).reset_index()
    fig_s, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_suppliers, x="Inventory_Value", y="Supplier", palette="copper", ax=ax)
    ax.set_title("Top 5 Suppliers by Inventory Value")
    ax.set_xlabel("Total Inventory Value ($)")
    save_matplotlib_fig(fig_s, "top_suppliers.png", output_dir)
    
    fig_s_ly = px.bar(top_suppliers, x="Inventory_Value", y="Supplier", orientation='h',
                       title="Top 5 Suppliers by Inventory Value",
                       labels={'Inventory_Value': 'Inventory Value ($)', 'Supplier': 'Supplier'})
    fig_s_ly.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig_p_ly, fig_c_ly, fig_s_ly

def run_eda_pipeline(raw_csv_path, engineered_csv_path, output_dir):
    print("Running Exploratory Data Analysis (EDA) Pipeline...")
    
    # Load files
    raw_df = pd.read_csv(raw_csv_path)
    eng_df = pd.read_csv(engineered_csv_path)
    
    # Ensure Date type parsed
    raw_df["Date_Added"] = pd.to_datetime(raw_df["Date_Added"], errors="coerce")
    eng_df["Date_Added"] = pd.to_datetime(eng_df["Date_Added"])
    if "Expiry_Date" in eng_df.columns:
        eng_df["Expiry_Date"] = pd.to_datetime(eng_df["Expiry_Date"])
        
    # Generate and save all plots
    generate_missing_values_chart(raw_df, output_dir)
    generate_correlation_matrix(eng_df, output_dir)
    generate_category_analysis(eng_df, output_dir)
    generate_brand_analysis(eng_df, output_dir)
    generate_supplier_analysis(eng_df, output_dir)
    generate_trends(eng_df, output_dir)
    generate_expiry_trend(eng_df, output_dir)
    generate_top_rankings(eng_df, output_dir)
    
    print(f"EDA plots saved successfully to: {output_dir}")

if __name__ == "__main__":
    src_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(src_dir)
    
    raw_csv = os.path.join(project_dir, "data", "raw", "SmartInventory_AI_Raw_Dataset.csv")
    engineered_csv = os.path.join(project_dir, "data", "processed", "SmartInventory_AI_Feature_Engineered.csv")
    figures_dir = os.path.join(project_dir, "reports", "figures")
    
    # Run pipeline
    run_eda_pipeline(raw_csv, engineered_csv, figures_dir)
