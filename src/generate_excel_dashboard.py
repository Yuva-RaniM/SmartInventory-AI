import os
import pandas as pd
import xlsxwriter

def create_excel_dashboard(data_path, output_path):
    print(f"Generating Executive Excel Dashboard from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source data for Excel dashboard not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Calculate aggregate KPI values
    total_products = len(df)
    total_inv_value = df["Inventory_Value"].sum()
    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_rev_at_risk = df["Revenue_at_Risk"].sum()
    total_recoverable = df["Recoverable_Revenue"].sum()
    avg_health = df["Inventory_Health"].mean()
    
    near_expiry = len(df[df["Expiry_Risk"] == "High"])
    low_stock = len(df[df["Stock_Status"] == "Low Stock"])
    overstock = len(df[df["Stock_Status"] == "Overstock"])
    
    # Grouped data for tables and charts
    category_summary = df.groupby("Category").agg(
        Product_Count=("Product_ID", "count"),
        Inventory_Value=("Inventory_Value", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Avg_Health=("Inventory_Health", "mean")
    ).reset_index()
    
    # Sort category summary by Revenue
    category_summary = category_summary.sort_values(by="Revenue", ascending=False)
    
    # Initialize Workbook
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    workbook = xlsxwriter.Workbook(output_path)
    
    # Define styles and palettes (Navy & Corporate palette)
    navy_header_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 18,
        'font_color': '#FFFFFF', 'bg_color': '#1B365D',
        'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#1B365D'
    })
    
    subtitle_fmt = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 11, 'italic': True,
        'font_color': '#FFFFFF', 'bg_color': '#1B365D',
        'align': 'center', 'valign': 'vcenter', 'border_color': '#1B365D'
    })
    
    kpi_title_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 9,
        'font_color': '#555555', 'bg_color': '#F4F5F7',
        'align': 'center', 'valign': 'vcenter',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    kpi_val_currency_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 14,
        'font_color': '#1B365D', 'bg_color': '#F4F5F7',
        'align': 'center', 'valign': 'vcenter', 'num_format': '₹#,##,##0',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    kpi_val_int_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 14,
        'font_color': '#1B365D', 'bg_color': '#F4F5F7',
        'align': 'center', 'valign': 'vcenter', 'num_format': '#,##0',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    kpi_val_pct_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 14,
        'font_color': '#1B365D', 'bg_color': '#F4F5F7',
        'align': 'center', 'valign': 'vcenter', 'num_format': '0.0"%"',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    # Red background for high risk values
    kpi_val_red_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 14,
        'font_color': '#9C0006', 'bg_color': '#FFC7CE',
        'align': 'center', 'valign': 'vcenter', 'num_format': '#,##0',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    table_header_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 10,
        'font_color': '#FFFFFF', 'bg_color': '#203A43',
        'align': 'center', 'valign': 'vcenter',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    table_data_fmt = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10,
        'align': 'left', 'valign': 'vcenter',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    table_data_num_fmt = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10,
        'align': 'right', 'valign': 'vcenter', 'num_format': '₹#,##,##0.00',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    table_data_pct_fmt = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10,
        'align': 'right', 'valign': 'vcenter', 'num_format': '0.0"%"',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    table_data_int_fmt = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10,
        'align': 'right', 'valign': 'vcenter', 'num_format': '#,##0',
        'border': 1, 'border_color': '#E1E4EA'
    })
    
    nav_btn_fmt = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 10,
        'font_color': '#1B365D', 'bg_color': '#D9E1F2',
        'align': 'center', 'valign': 'vcenter', 'border': 2, 'border_color': '#1B365D'
    })
    
    # -------------------------------------------------------------
    # SHEET 1: Executive Dashboard
    # -------------------------------------------------------------
    dash_sheet = workbook.add_worksheet("Executive Dashboard")
    dash_sheet.hide_gridlines(2) # Hide gridlines
    
    # Set Column Widths for nice margins
    dash_sheet.set_column("A:A", 3)   # Margin
    dash_sheet.set_column("B:D", 15)  # KPI Area
    dash_sheet.set_column("E:E", 4)   # Spacer
    dash_sheet.set_column("F:H", 16)  # Table Area
    dash_sheet.set_column("I:N", 15)  # Chart Area
    
    # Title Block (Merged A2:N3)
    dash_sheet.merge_range("B2:N2", "SMARTINVENTORY AI", navy_header_fmt)
    dash_sheet.merge_range("B3:N3", "Executive Revenue Recovery & Inventory Intelligence Dashboard", subtitle_fmt)
    dash_sheet.set_row(1, 25)
    dash_sheet.set_row(2, 18)
    
    # Navigation Link Button
    dash_sheet.write_url("B5", "internal:'Inventory Data'!A1", string="View Raw Inventory Data", cell_format=nav_btn_fmt)
    dash_sheet.set_row(4, 25)
    
    # --- Row 1 KPIs (Rows 7-9) ---
    # Col B:C (Total Products), Col D:E (Inventory Value), Col F:G (Projected Revenue), Col H:I (Expected Profit), Col J:K (Revenue at Risk)
    kpis = [
        # Label, Value, Format, ColStart, ColEnd, RowStart
        ("TOTAL PRODUCTS", total_products, kpi_val_int_fmt, "B", "B", 7),
        ("INVENTORY VALUE", total_inv_value, kpi_val_currency_fmt, "C", "C", 7),
        ("REVENUE", total_revenue, kpi_val_currency_fmt, "D", "D", 7),
        ("EXPECTED PROFIT", total_profit, kpi_val_currency_fmt, "E", "E", 7),
        ("REVENUE AT RISK", total_rev_at_risk, kpi_val_currency_fmt, "F", "F", 7),
    ]
    
    for label, val, val_fmt, start_col, end_col, start_row in kpis:
        dash_sheet.write(f"{start_col}{start_row}", label, kpi_title_fmt)
        dash_sheet.write(f"{start_col}{start_row+1}", val, val_fmt)
        dash_sheet.set_row(start_row - 1, 15)
        dash_sheet.set_row(start_row, 28)
        
    # --- Row 2 KPIs (Rows 10-12) ---
    kpis_r2 = [
        ("RECOVERABLE REVENUE", total_recoverable, kpi_val_currency_fmt, "B", "B", 10),
        ("INVENTORY HEALTH", avg_health, kpi_val_pct_fmt, "C", "C", 10),
        ("PRODUCTS NEAR EXPIRY", near_expiry, kpi_val_red_fmt if near_expiry > 0 else kpi_val_int_fmt, "D", "D", 10),
        ("LOW STOCK PRODUCTS", low_stock, kpi_val_red_fmt if low_stock > 0 else kpi_val_int_fmt, "E", "E", 10),
        ("OVERSTOCK PRODUCTS", overstock, kpi_val_int_fmt, "F", "F", 10),
    ]
    
    for label, val, val_fmt, start_col, end_col, start_row in kpis_r2:
        dash_sheet.write(f"{start_col}{start_row}", label, kpi_title_fmt)
        dash_sheet.write(f"{start_col}{start_row+1}", val, val_fmt)
        dash_sheet.set_row(start_row - 1, 15)
        dash_sheet.set_row(start_row, 28)
        
    # --- Category Breakdown Table (Rows 14+) ---
    start_r = 13
    dash_sheet.write(f"B{start_r}", "Category", table_header_fmt)
    dash_sheet.write(f"C{start_r}", "Products Count", table_header_fmt)
    dash_sheet.write(f"D{start_r}", "Inventory Value", table_header_fmt)
    dash_sheet.write(f"E{start_r}", "Revenue", table_header_fmt)
    dash_sheet.write(f"F{start_r}", "Avg Health", table_header_fmt)
    
    for idx, row in category_summary.iterrows():
        r = start_r + 1 + idx
        dash_sheet.write(f"B{r}", row["Category"], table_data_fmt)
        dash_sheet.write(f"C{r}", row["Product_Count"], table_data_int_fmt)
        dash_sheet.write(f"D{r}", row["Inventory_Value"], table_data_num_fmt)
        dash_sheet.write(f"E{r}", row["Revenue"], table_data_num_fmt)
        dash_sheet.write(f"F{r}", row["Avg_Health"], table_data_pct_fmt)
        
    # Add native Excel Chart 1: Revenue vs Inventory by Category
    chart1 = workbook.add_chart({'type': 'column'})
    chart1.add_series({
        'name': 'Inventory Value',
        'categories': f"='Executive Dashboard'!$B$14:$B${14 + len(category_summary)-1}",
        'values': f"='Executive Dashboard'!$D$14:$D${14 + len(category_summary)-1}",
        'fill': {'color': '#1F4E79'}
    })
    chart1.add_series({
        'name': 'Revenue',
        'categories': f"='Executive Dashboard'!$B$14:$B${14 + len(category_summary)-1}",
        'values': f"='Executive Dashboard'!$E$14:$E${14 + len(category_summary)-1}",
        'fill': {'color': '#ED7D31'}
    })
    chart1.set_title({'name': 'Category Revenue & Capital Comparison', 'name_font': {'name': 'Segoe UI', 'size': 12, 'bold': True}})
    chart1.set_x_axis({'name': 'Category', 'name_font': {'name': 'Segoe UI', 'size': 9}})
    chart1.set_y_axis({'name': 'Amount (₹)', 'name_font': {'name': 'Segoe UI', 'size': 9}})
    chart1.set_legend({'position': 'bottom'})
    chart1.set_size({'width': 500, 'height': 300})
    dash_sheet.insert_chart('H14', chart1)
    
    # -------------------------------------------------------------
    # SHEET 2: Inventory Data (Raw + Features)
    # -------------------------------------------------------------
    data_sheet = workbook.add_worksheet("Inventory Data")
    
    # Write dataframe with standard style
    cols = list(df.columns)
    for c_idx, col in enumerate(cols):
        data_sheet.write(0, c_idx, col, table_header_fmt)
        
    # Write rows
    for r_idx, row in df.iterrows():
        for c_idx, col in enumerate(cols):
            val = row[col]
            if pd.isna(val):
                data_sheet.write(r_idx+1, c_idx, "", table_data_fmt)
                continue
            # Formats based on data types
            if isinstance(val, float):
                data_sheet.write(r_idx+1, c_idx, val, table_data_num_fmt if "Value" in col or "Revenue" in col or "Profit" in col or "Price" in col else table_data_fmt)
            elif isinstance(val, int):
                data_sheet.write(r_idx+1, c_idx, val, table_data_int_fmt)
            else:
                data_sheet.write(r_idx+1, c_idx, str(val), table_data_fmt)
                
    # Auto-fit column widths
    data_sheet.autofit()
    
    # Save file
    workbook.close()
    print(f"Executive Excel dashboard successfully built and saved to: {output_path}")

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    eng_csv = os.path.join(project_dir, "data", "processed", "SmartInventory_AI_Feature_Engineered.csv")
    excel_path = os.path.join(project_dir, "dashboards", "excel", "SmartInventory_AI_Executive_Dashboard.xlsx")
    
    create_excel_dashboard(eng_csv, excel_path)
