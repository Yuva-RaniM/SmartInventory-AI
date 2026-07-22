import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import project modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import remove_duplicates, handle_missing_values, remove_invalid_records, standardize_strings, convert_date_columns, validate_business_rules, convert_data_types
from src.feature_engineering import run_feature_engineering
from src.recommendation_engine import generate_recommendations
from src.prediction import run_predictions
from src.train_models import train_and_save_models
from src.generate_excel_dashboard import create_excel_dashboard
from src.database import get_db_engine, init_db, load_data_to_db
from src.schema_adapter import standardize_dataset
from src.ai_assistant import answer as ai_answer

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Configure Streamlit page layout
st.set_page_config(
    page_title="SmartInventory AI – Revenue Recovery Console",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#e8eef6; --muted:#94a3b8; --panel:rgba(17,24,39,.86); --line:rgba(148,163,184,.18); --emerald:#10b981; --gold:#f59e0b; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp {
        color:var(--ink);
        background:
          radial-gradient(circle at 15% 10%, rgba(16,185,129,.14), transparent 30%),
          radial-gradient(circle at 90% 25%, rgba(245,158,11,.10), transparent 26%),
          linear-gradient(rgba(3,7,18,.94),rgba(3,7,18,.97)),
          url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='700'%3E%3Cg fill='none' stroke='%2310b981' stroke-opacity='.14'%3E%3Cpath d='M0 560h1200M90 560V250h300v310M470 560V160h250v400M810 560V300h300v260'/%3E%3Cpath d='M120 300h240M120 370h240M120 440h240M500 220h190M500 300h190M500 380h190M840 350h240M840 430h240'/%3E%3C/g%3E%3C/svg%3E") center/cover fixed;
    }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stAppViewContainer"] > .main .block-container { max-width:1480px; padding-top:1.5rem; }
    section[data-testid="stSidebar"] { background:rgba(6,11,20,.96); border-right:1px solid var(--line); }
    section[data-testid="stSidebar"] * { color:#e5e7eb; }
    section[data-testid="stSidebar"] [role="radiogroup"] label { border-radius:11px; padding:.34rem .55rem; transition:.2s ease; }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(16,185,129,.12); transform:translateX(3px); }
    h1,h2,h3 { color:#f8fafc; letter-spacing:-.025em; }
    p, label, [data-testid="stMarkdownContainer"] { color:#cbd5e1; }
    [data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], div[data-testid="stExpander"] {
        background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:.7rem; box-shadow:0 18px 50px rgba(0,0,0,.22); backdrop-filter:blur(14px);
    }
    .kpi-card {
        background:linear-gradient(145deg,rgba(31,41,55,.92),rgba(17,24,39,.88)); padding:1.35rem; border-radius:16px;
        box-shadow:0 18px 45px rgba(0,0,0,.24); border:1px solid var(--line); border-top:3px solid #6366f1; margin-bottom:1rem; transition:.25s ease;
    }
    .kpi-card:hover { transform:translateY(-3px); border-color:rgba(16,185,129,.42); }
    .kpi-card.risk { border-top-color:#ef4444; } .kpi-card.recovery { border-top-color:#10b981; } .kpi-card.health { border-top-color:#f59e0b; }
    .kpi-val { font-size:1.62rem; font-weight:800; color:#f8fafc; margin-top:7px; }
    .kpi-label { font-size:.74rem; text-transform:uppercase; font-weight:700; color:#94a3b8; letter-spacing:.09em; }
    .glass-header {
        background:linear-gradient(120deg,rgba(17,24,39,.94),rgba(6,78,59,.62)); padding:2.7rem; border-radius:22px;
        color:white; margin-bottom:2rem; border:1px solid rgba(16,185,129,.28); box-shadow:0 24px 70px rgba(0,0,0,.28);
    }
    .glass-header h1 { color:white; margin:0; font-size:2.35rem; } .glass-header p { color:#a7f3d0; margin-top:.65rem; font-size:1.05rem; }
    .stButton>button {
        background:linear-gradient(135deg,#059669,#10b981); color:#fff; border:1px solid rgba(255,255,255,.12);
        padding:.62rem 1.35rem; border-radius:11px; font-weight:700; transition:all .22s ease;
    }
    .stButton>button:hover { transform:translateY(-2px); box-shadow:0 9px 25px rgba(16,185,129,.26); border-color:#34d399; }
    .login-shell { max-width:470px; margin:5vh auto 1rem; padding:2rem 2.1rem; border-radius:24px; background:rgba(10,17,29,.9); border:1px solid rgba(16,185,129,.25); box-shadow:0 35px 90px rgba(0,0,0,.45); text-align:center; }
    .login-mark { width:56px; height:56px; margin:auto; border-radius:17px; display:grid; place-items:center; font-weight:800; font-size:1.25rem; color:#03130d; background:linear-gradient(135deg,#34d399,#f59e0b); }
    .login-shell h1 { margin:.9rem 0 .3rem; } .login-shell p { color:#94a3b8; }
    [data-testid="stChatMessage"] { background:rgba(17,24,39,.74); border:1px solid var(--line); border-radius:16px; padding:.6rem 1rem; }
</style>
""", unsafe_allow_html=True)

# Authentication is intentionally environment-driven. No API key is exposed in UI.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <style>section[data-testid="stSidebar"]{display:none}</style>
    <div class="login-shell"><div class="login-mark">SI</div><h1>SmartInventory AI</h1>
    <p>Inventory decision intelligence for revenue protection, demand planning and expiry-risk control.</p></div>
    """, unsafe_allow_html=True)
    left_space, login_column, right_space = st.columns([1.15, 1, 1.15])
    with login_column:
        with st.form("secure_login", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in securely", use_container_width=True)
    if submitted:
        valid_user = os.getenv("APP_USERNAME", "admin")
        valid_password = os.getenv("APP_PASSWORD", "ChangeMe123!")
        if username == valid_user and password == valid_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Username or password is incorrect.")
    st.caption("SmartInventory AI · Enterprise Edition · v2.1")
    st.stop()

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "SmartInventory_AI_Raw_Dataset.csv")
CLEANED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "SmartInventory_AI_Cleaned_Dataset.csv")
ENGINEERED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "SmartInventory_AI_Feature_Engineered.csv")
PREDICTION_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Prediction.csv")
RECOMMENDATION_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Recommendation_Report.csv")
SUMMARY_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Business_Impact_Summary.json")
EXCEL_PATH = os.path.join(PROJECT_ROOT, "dashboards", "excel", "SmartInventory_AI_Executive_Dashboard.xlsx")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "SmartInventory_AI_Feature_Engineered.csv"
)

@st.cache_data
def load_inventory_data() -> pd.DataFrame:
    """
    Load and prepare the SmartInventory dataset.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}"
        )

    dataframe = pd.read_csv(
        DATA_FILE
    )

    # Date columns
    date_columns = [
        "Manufacturing_Date",
        "Expiry_Date",
        "Timestamp"
    ]

    for column in date_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce"
            )

    # Numeric columns
    numeric_columns = [
        "Days_Remaining",
        "Purchase_Price",
        "Selling_Price",
        "Current_Stock",
        "Reorder_Level",
        "Daily_Sales",
        "Weekly_Demand",
        "Monthly_Demand",
        "Lead_Time_Days",
        "Demand_Forecast",
        "Revenue_At_Risk",
        "Discount_%",
        "Customer_Rating",
        "Waste_Cost",
        "Recovered_Revenue"
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            ).fillna(0)

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    dataframe["Inventory_Value"] = (
        dataframe["Current_Stock"]
        * dataframe["Purchase_Price"]
    )

    dataframe["Potential_Sales_Value"] = (
        dataframe["Current_Stock"]
        * dataframe["Selling_Price"]
    )

    dataframe["Estimated_Daily_Revenue"] = (
        dataframe["Daily_Sales"]
        * dataframe["Selling_Price"]
    )

    dataframe["Estimated_Daily_Profit"] = (
        dataframe["Daily_Sales"]
        * (
            dataframe["Selling_Price"]
            - dataframe["Purchase_Price"]
        )
    )

    dataframe["Profit_Margin_%"] = np.where(
        dataframe["Selling_Price"] > 0,
        (
            (
                dataframe["Selling_Price"]
                - dataframe["Purchase_Price"]
            )
            / dataframe["Selling_Price"]
        ) * 100,
        0
    )

    dataframe["Recovery_Rate_%"] = np.where(
        dataframe["Revenue_At_Risk"] > 0,
        (
            dataframe["Recovered_Revenue"]
            / dataframe["Revenue_At_Risk"]
        ) * 100,
        0
    )

    dataframe["Stock_Coverage_Days"] = np.where(
        dataframe["Daily_Sales"] > 0,
        (
            dataframe["Current_Stock"]
            / dataframe["Daily_Sales"]
        ),
        0
    )

    return dataframe

try:
    df = load_inventory_data()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

# Sidebar navigation logo and header
st.sidebar.markdown("<h2 style='text-align:center'>SmartInventory AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center;color:#94a3b8'>Decision Intelligence Workspace</p>", unsafe_allow_html=True)
st.sidebar.divider()

# Pages list
PAGE_OPTIONS = [
    "🏠 Home",
    "📤 Upload Dataset",
    "📋 Data Validation",
    "🧹 Data Cleaning",
    "⚙️ Feature Engineering",
    "📊 EDA (Analysis)",
    "🤖 Run AI Models",
    "📈 Prediction Results",
    "💡 AI Recommendations",
    "🖥️ Executive Dashboard",
    "📥 Download Reports"
    ,"💬 AI Business Assistant"
]

page = st.sidebar.radio("Workspace", PAGE_OPTIONS)
theme_mode = st.sidebar.selectbox("Appearance", ["Dark", "Light"], key="theme_mode")
px.defaults.template = "plotly_white" if theme_mode == "Light" else "plotly_dark"
if theme_mode == "Light":
    st.markdown("""
    <style>
    .stApp{color:#172033;background:radial-gradient(circle at 12% 10%,rgba(16,185,129,.13),transparent 28%),linear-gradient(rgba(248,250,252,.95),rgba(241,245,249,.97)),url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='700'%3E%3Cg fill='none' stroke='%23059669' stroke-opacity='.12'%3E%3Cpath d='M0 560h1200M90 560V250h300v310M470 560V160h250v400M810 560V300h300v260'/%3E%3C/g%3E%3C/svg%3E") center/cover fixed}
    h1,h2,h3{color:#111827} p,label,[data-testid="stMarkdownContainer"]{color:#334155}
    [data-testid="stMetric"],[data-testid="stDataFrame"],[data-testid="stPlotlyChart"],div[data-testid="stExpander"]{background:rgba(255,255,255,.88);border-color:rgba(15,23,42,.12)}
    .kpi-card{background:linear-gradient(145deg,#fff,#f8fafc)} .kpi-val{color:#111827}
    </style>""", unsafe_allow_html=True)
ai_ready = bool(os.getenv("GEMINI_API_KEY", "").strip())
st.sidebar.caption(f"AI connection: {'Ready' if ai_ready else 'Add key in .env'}")
if st.sidebar.button("Sign out", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

# Helper function to load datasets into session state if they exist on disk
def bootstrap_session_state():
    if "raw_df" not in st.session_state and os.path.exists(RAW_CSV_PATH):
        st.session_state["raw_df"] = pd.read_csv(RAW_CSV_PATH)
    if "cleaned_df" not in st.session_state and os.path.exists(CLEANED_CSV_PATH):
        st.session_state["cleaned_df"] = pd.read_csv(CLEANED_CSV_PATH)
    if "engineered_df" not in st.session_state and os.path.exists(ENGINEERED_CSV_PATH):
        st.session_state["engineered_df"] = pd.read_csv(ENGINEERED_CSV_PATH)
    if "prediction_df" not in st.session_state and os.path.exists(PREDICTION_CSV_PATH):
        st.session_state["prediction_df"] = pd.read_csv(PREDICTION_CSV_PATH)
    if "recs_df" not in st.session_state and os.path.exists(RECOMMENDATION_CSV_PATH):
        st.session_state["recs_df"] = pd.read_csv(RECOMMENDATION_CSV_PATH)
    if "business_summary" not in st.session_state and os.path.exists(SUMMARY_JSON_PATH):
        with open(SUMMARY_JSON_PATH, "r") as f:
            st.session_state["business_summary"] = json.load(f)

bootstrap_session_state()

# -------------------------------------------------------------
# PAGE 1: Home / Login
# -------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <div class="glass-header">
        <h1>SmartInventory AI</h1>
        <p>AI-Powered Revenue Recovery and Inventory Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Welcome to SmartInventory AI")
        st.markdown("""
        SmartInventory AI is an **enterprise-grade Decision Support System (DSS)** designed to protect retail, pharmaceutical, and grocery operations from revenue decay.
        
        The system automatically scans stock profiles, predicts expiry risk, forecasts future demands, and formulates recovery decisions such as markdowns, stock transfers, and reorder volumes.
        
        ### Workflow Pipeline:
        1. **Data Preprocessing**: Strips anomalies, duplicate entities, and enforces business rules.
        2. **Feature Augmentation**: Generates domain specific metrics (ABC categories, Turnover, Days to Expiry, Inventory Values).
        3. **Exploratory Data Analysis (EDA)**: Direct visual evaluation of trends and correlations.
        4. **AI-Driven Inference**: Uses models (Random Forest, XGBoost, Linear Regression, K-Means Clustering) to predict risks, segment stock performance, and project future revenues.
        5. **Prescriptive Recommendations**: Recommends strategic decisions (Discounts, Transfers, Reorders) and estimates financial recovery values.
        6. **Multi-Channel Dashboards**: Generates dynamic Excel dashboards, provides Power BI design blueprints, and runs SQLite/MySQL persistence.
        """)
        
    with col2:
        st.subheader("System Readiness")
        st.metric("Dataset records", f"{len(df):,}")
        st.metric("AI assistant", "Connected" if ai_ready else "Key required")
        st.metric("Saved ML models", f"{sum(os.path.exists(os.path.join(MODELS_DIR, name)) for name in ['expiry_model.pkl','demand_model.pkl','revenue_model.pkl','cluster_model.pkl'])}/4")

        # Load sample synthetic database helper
        if not os.path.exists(RAW_CSV_PATH):
            st.warning("No dataset detected on disk.")
            if st.button("Bootstrap Synthetic Raw Dataset"):
                from src.generate_synthetic_data import generate_raw_dataset
                generate_raw_dataset(RAW_CSV_PATH, num_records=50000)
                st.session_state["raw_df"] = pd.read_csv(RAW_CSV_PATH)
                st.rerun()
        else:
            st.info("Sample synthetic raw data is available.")

# -------------------------------------------------------------
# PAGE 2: Upload Dataset
# -------------------------------------------------------------
elif page == "📤 Upload Dataset":
    st.title("📤 Upload Inventory Dataset")
    st.markdown("Upload your store's raw inventory CSV file to initiate the AI preprocessing and analysis pipeline.")
    
    uploaded_file = st.file_uploader("Choose inventory CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        source_df = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
        raw_df, schema_report = standardize_dataset(source_df)
        st.session_state["raw_df"] = raw_df
        st.session_state["source_df"] = source_df
        st.session_state["schema_report"] = schema_report
        # Save to raw path for processing
        os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)
        raw_df.to_csv(RAW_CSV_PATH, index=False)
        st.success(f"Successfully uploaded dataset: {uploaded_file.name} (Shape: {raw_df.shape})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Schema Compatibility", f"{schema_report.compatibility_score:.1f}%")
        c2.metric("Mapped Fields", len(schema_report.mappings))
        c3.metric("Supported Analyses", len(schema_report.supported_analyses))
        with st.expander("Standard schema mapping report", expanded=True):
            mapping_rows = [{"Canonical Field": k, "Source Column": v, "Method": schema_report.methods.get(k)} for k, v in schema_report.mappings.items()]
            st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)
            st.write("**Supported:**", ", ".join(schema_report.supported_analyses) or "profiling only")
            st.write("**Derived fields:**", ", ".join(schema_report.derived_fields) or "None")
            if schema_report.warnings:
                for warning in schema_report.warnings: st.warning(warning)
        
        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head(10))
    else:
        if "raw_df" in st.session_state:
            st.info("Using currently loaded sample dataset.")
            st.dataframe(st.session_state["raw_df"].head(10))
            if st.button("Reload Sample Raw Dataset"):
                from src.generate_synthetic_data import generate_raw_dataset
                generate_raw_dataset(RAW_CSV_PATH, num_records=5000)
                st.session_state["raw_df"] = pd.read_csv(RAW_CSV_PATH)
                st.rerun()
        else:
            st.warning("No dataset loaded. Please upload a file or click the button on the Home page to generate sample data.")

# -------------------------------------------------------------
# PAGE 3: Data Validation
# -------------------------------------------------------------
elif page == "📋 Data Validation":
    st.title("📋 Data Validation")
    st.markdown("Validates integrity thresholds and highlights structural anomalies present in the raw data.")
    
    if "raw_df" not in st.session_state:
        st.warning("Please upload or generate a dataset first.")
    else:
        df = st.session_state["raw_df"]
        
        st.subheader("Data Profile Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Row Count", f"{len(df):,}")
        col2.metric("Total Attribute Fields", len(df.columns))
        col3.metric("Null Cell Percentage", f"{(df.isnull().sum().sum() / df.size * 100):.2f}%")
        
        # Missing values breakdown
        st.subheader("Null Attribute Distribution")
        null_counts = df.isnull().sum()
        null_df = pd.DataFrame({"Missing Values": null_counts, "Percentage (%)": (null_counts / len(df) * 100).round(2)})
        null_df = null_df[null_df["Missing Values"] > 0]
        
        if len(null_df) > 0:
            st.dataframe(null_df)
        else:
            st.success("No null values detected in raw structure!")
            
        # Business rules validation check
        st.subheader("Business Rules Compliance Report")
        anomalies = []
        
        # 1. Duplicates check
        dups = df.duplicated().sum()
        if dups > 0:
            anomalies.append(f"• **Duplicates**: {dups} exact duplicate records found.")
            
        # 2. Negative stock
        if "Stock_Quantity" in df.columns:
            neg_stock = (df["Stock_Quantity"] < 0).sum()
            if neg_stock > 0:
                anomalies.append(f"• **Negative Stock**: {neg_stock} records have negative inventory levels.")
                
        # 3. Negative price
        if "Purchase_Price" in df.columns:
            neg_cost = (df["Purchase_Price"] <= 0).sum()
            if neg_cost > 0:
                anomalies.append(f"• **Negative/Zero Cost**: {neg_cost} records have cost price <= 0.")
                
        # 4. Markup violation
        if "Purchase_Price" in df.columns and "Selling_Price" in df.columns:
            markup_viol = (df["Selling_Price"] < df["Purchase_Price"]).sum()
            if markup_viol > 0:
                anomalies.append(f"• **Profit Margin Violation**: {markup_viol} records have Selling_Price < Cost_Price.")
                
        if anomalies:
            st.error("Structural Anomalies Detected:")
            for anomaly in anomalies:
                st.markdown(anomaly)
            st.info("Navigate to the 'Data Cleaning' page to run the automated remediation pipeline.")
        else:
            st.success("All data passed core business rule validation!")

# -------------------------------------------------------------
# PAGE 4: Data Cleaning
# -------------------------------------------------------------
elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Preprocessing & Cleaning")
    st.markdown("Removes duplicates, imputes missing values, corrects negative prices/quantities, and standardizes casing.")
    
    if "raw_df" not in st.session_state:
        st.warning("Please upload or generate a dataset first.")
    else:
        raw_df = st.session_state["raw_df"]
        
        if st.button("Run Preprocessing Pipeline"):
            with st.spinner("Executing data cleaning algorithms..."):
                # Run the step-by-step logic
                cleaned_df = remove_duplicates(raw_df)
                cleaned_df = handle_missing_values(cleaned_df)
                cleaned_df = remove_invalid_records(cleaned_df)
                cleaned_df = standardize_strings(cleaned_df)
                cleaned_df = convert_date_columns(cleaned_df)
                cleaned_df = validate_business_rules(cleaned_df)
                cleaned_df = convert_data_types(cleaned_df)
                
                # Save to disk
                os.makedirs(os.path.dirname(CLEANED_CSV_PATH), exist_ok=True)
                cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
                st.session_state["cleaned_df"] = cleaned_df
                
                st.success("Cleaning Pipeline completed successfully!")
                
        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]
            
            st.subheader("Preprocessing Results Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw Dataset Profile**")
                st.write(f"• Rows: {len(raw_df)}")
                st.write(f"• Columns: {len(raw_df.columns)}")
                st.write(f"• Missing cells: {raw_df.isnull().sum().sum()}")
            with col2:
                st.markdown("**Cleaned Dataset Profile**")
                st.write(f"• Rows: {len(cleaned_df)}")
                st.write(f"• Columns: {len(cleaned_df.columns)}")
                st.write(f"• Missing cells: {cleaned_df.isnull().sum().sum()}")
                
            st.subheader("Cleaned Dataset Preview")
            st.dataframe(cleaned_df.head(10))

# -------------------------------------------------------------
# PAGE 5: Feature Engineering
# -------------------------------------------------------------
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")
    st.markdown("Generates domain specific metrics (ABC categories, Inventory Turnover, Days to Expiry, Profit margins).")
    
    if "cleaned_df" not in st.session_state:
        st.warning("Please run the Data Preprocessing/Cleaning page first.")
    else:
        cleaned_df = st.session_state["cleaned_df"]
        
        if st.button("Generate Augmented Features"):
            with st.spinner("Calculating business formulas and risk profiles..."):
                engineered_df = run_feature_engineering(CLEANED_CSV_PATH, ENGINEERED_CSV_PATH)
                st.session_state["engineered_df"] = engineered_df
                st.success("Feature Engineering complete! New feature metrics appended.")
                
        if "engineered_df" in st.session_state:
            engineered_df = st.session_state["engineered_df"]
            
            st.subheader("Feature Engineered Preview")
            st.dataframe(engineered_df.head(10))
            

# -------------------------------------------------------------
# PAGE 6: EDA (Exploratory Data Analysis)
# -------------------------------------------------------------
elif page == "📊 EDA (Analysis)":

    # -----------------------------------------------------
    # PAGE HEADER
    # -----------------------------------------------------

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown(
        """
        Interactive analytical charts for categories, brands,
        suppliers, historical trends, expiry risks and sales performance.
        """
    )

    # Work with a separate copy
    eda_df = df.copy()

    # -----------------------------------------------------
    # SIDEBAR FILTERS
    # -----------------------------------------------------

    st.sidebar.markdown("## EDA Filters")

    category_options = sorted(
        eda_df["Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "Select Category",
        options=category_options,
        default=category_options
    )

    brand_options = sorted(
        eda_df["Brand"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_brands = st.sidebar.multiselect(
        "Select Brand",
        options=brand_options,
        default=brand_options
    )

    supplier_options = sorted(
        eda_df["Supplier"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_suppliers = st.sidebar.multiselect(
        "Select Supplier",
        options=supplier_options,
        default=supplier_options
    )

    expiry_options = sorted(
        eda_df["Expiry_Risk"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_expiry_risks = st.sidebar.multiselect(
        "Select Expiry Risk",
        options=expiry_options,
        default=expiry_options
    )

    # Apply filters
    filtered_df = eda_df[
        eda_df["Category"].astype(str).isin(selected_categories)
        & eda_df["Brand"].astype(str).isin(selected_brands)
        & eda_df["Supplier"].astype(str).isin(selected_suppliers)
        & eda_df["Expiry_Risk"].astype(str).isin(
            selected_expiry_risks
        )
    ].copy()

    if filtered_df.empty:
        st.warning(
            "No records available for the selected filters."
        )
        st.stop()

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    total_products = filtered_df["Product_ID"].nunique()

    total_inventory_value = filtered_df[
        "Inventory_Value"
    ].sum()

    total_revenue_at_risk = filtered_df[
        "Revenue_At_Risk"
    ].sum()

    total_recovered_revenue = filtered_df[
        "Recovered_Revenue"
    ].sum()

    average_rating = filtered_df[
        "Customer_Rating"
    ].mean()

    average_daily_sales = filtered_df[
        "Daily_Sales"
    ].mean()

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(
            "Total Products",
            f"{total_products:,}"
        )

    with kpi2:
        st.metric(
            "Inventory Value",
            f"₹{total_inventory_value:,.0f}"
        )

    with kpi3:
        st.metric(
            "Revenue at Risk",
            f"₹{total_revenue_at_risk:,.0f}"
        )

    kpi4, kpi5, kpi6 = st.columns(3)

    with kpi4:
        st.metric(
            "Recovered Revenue",
            f"₹{total_recovered_revenue:,.0f}"
        )

    with kpi5:
        st.metric(
            "Average Customer Rating",
            f"{average_rating:.2f} / 5"
        )

    with kpi6:
        st.metric(
            "Average Daily Sales",
            f"{average_daily_sales:.2f}"
        )

    st.divider()

    # -----------------------------------------------------
    # CHART SELECTION
    # -----------------------------------------------------

    chart_options = [
        "Product Category Analysis",
        "Correlation Heatmap",
        "Brand Value Distribution",
        "Supplier Contribution Analysis",
        "Historical Trends (Revenue & Inventory)",
        "Expiry Timelines & Capital At Risk",
        "Sales & Performance Rankings"
    ]

    selected_chart = st.selectbox(
        "Select Chart Category",
        options=chart_options
    )

    # =====================================================
    # 1. PRODUCT CATEGORY ANALYSIS
    # =====================================================

    if selected_chart == "Product Category Analysis":

        st.subheader("📦 Product Category Analysis")

        category_summary = (
            filtered_df
            .groupby(
                "Category",
                as_index=False
            )
            .agg(
                Inventory_Value=(
                    "Inventory_Value",
                    "sum"
                ),
                Current_Stock=(
                    "Current_Stock",
                    "sum"
                ),
                Daily_Sales=(
                    "Daily_Sales",
                    "sum"
                ),
                Revenue_At_Risk=(
                    "Revenue_At_Risk",
                    "sum"
                )
            )
            .sort_values(
                "Inventory_Value",
                ascending=False
            )
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "Inventory Value",
                "Current Stock",
                "Category Share"
            ]
        )

        with tab1:

            fig = px.bar(
                category_summary,
                x="Category",
                y="Inventory_Value",
                color="Inventory_Value",
                text_auto=True,
                title="Inventory Value by Product Category",
                labels={
                    "Inventory_Value": "Inventory Value (₹)"
                }
            )

            fig.update_traces(
                texttemplate="%{y:,.0f}",
                textposition="auto"
            )

            fig.update_layout(
                xaxis_title="Product Category",
                yaxis_title="Inventory Value (₹)",
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab2:

            fig = px.bar(
                category_summary,
                x="Category",
                y=[
                    "Current_Stock",
                    "Daily_Sales"
                ],
                barmode="group",
                title=(
                    "Current Stock vs Daily Sales "
                    "by Category"
                ),
                labels={
                    "value": "Quantity",
                    "variable": "Metric"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab3:

            fig = px.pie(
                category_summary,
                names="Category",
                values="Inventory_Value",
                hole=0.45,
                title="Category-wise Inventory Value Share"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            category_summary,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 2. CORRELATION HEATMAP
    # =====================================================

    elif selected_chart == "Correlation Heatmap":

        st.subheader("🔗 Correlation Heatmap")

        correlation_columns = [
            "Purchase_Price",
            "Selling_Price",
            "Current_Stock",
            "Reorder_Level",
            "Daily_Sales",
            "Weekly_Demand",
            "Monthly_Demand",
            "Lead_Time_Days",
            "Demand_Forecast",
            "Revenue_At_Risk",
            "Discount_%",
            "Customer_Rating",
            "Waste_Cost",
            "Recovered_Revenue",
            "Inventory_Value",
            "Estimated_Daily_Revenue",
            "Estimated_Daily_Profit",
            "Profit_Margin_%"
        ]

        available_columns = [
            column
            for column in correlation_columns
            if column in filtered_df.columns
        ]

        selected_correlation_columns = st.multiselect(
            "Select numerical features",
            options=available_columns,
            default=available_columns[:12]
        )

        if len(selected_correlation_columns) < 2:

            st.warning(
                "Select at least two numerical columns."
            )

        else:

            correlation_matrix = (
                filtered_df[
                    selected_correlation_columns
                ]
                .corr()
                .round(2)
            )

            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Matrix"
            )

            fig.update_layout(
                height=750
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                """
                Values close to +1 indicate a strong positive
                relationship. Values close to -1 indicate a strong
                negative relationship.
                """
            )

    # =====================================================
    # 3. BRAND VALUE DISTRIBUTION
    # =====================================================

    elif selected_chart == "Brand Value Distribution":

        st.subheader("🏷️ Brand Value Distribution")

        top_brand_count = st.slider(
            "Number of brands to display",
            min_value=5,
            max_value=30,
            value=15,
            step=1
        )

        brand_summary = (
            filtered_df
            .groupby(
                "Brand",
                as_index=False
            )
            .agg(
                Inventory_Value=(
                    "Inventory_Value",
                    "sum"
                ),
                Current_Stock=(
                    "Current_Stock",
                    "sum"
                ),
                Estimated_Daily_Revenue=(
                    "Estimated_Daily_Revenue",
                    "sum"
                ),
                Revenue_At_Risk=(
                    "Revenue_At_Risk",
                    "sum"
                ),
                Product_Count=(
                    "Product_ID",
                    "nunique"
                )
            )
            .sort_values(
                "Inventory_Value",
                ascending=False
            )
            .head(top_brand_count)
        )

        tab1, tab2 = st.tabs(
            [
                "Brand Inventory Value",
                "Revenue and Risk"
            ]
        )

        with tab1:

            fig = px.bar(
                brand_summary.sort_values("Inventory_Value"),
                x="Inventory_Value",
                y="Brand",
                orientation="h",
                color="Inventory_Value",
                text_auto=True,
                title=(
                    f"Top {top_brand_count} Brands "
                    "by Inventory Value"
                ),
                labels={
                    "Inventory_Value": "Inventory Value (₹)"
                }
            )

            fig.update_traces(
                texttemplate="%{x:,.0f}",
                textposition="auto"
            )

            fig.update_layout(
                height=max(
                    500,
                    top_brand_count * 35
                ),
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab2:

            fig = px.scatter(
                brand_summary,
                x="Estimated_Daily_Revenue",
                y="Revenue_At_Risk",
                size="Inventory_Value",
                color="Brand",
                hover_name="Brand",
                title=(
                    "Brand Daily Revenue vs "
                    "Revenue at Risk"
                ),
                labels={
                    "Estimated_Daily_Revenue":
                        "Estimated Daily Revenue (₹)",
                    "Revenue_At_Risk":
                        "Revenue at Risk (₹)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            brand_summary,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 4. SUPPLIER CONTRIBUTION ANALYSIS
    # =====================================================

    elif selected_chart == "Supplier Contribution Analysis":

        st.subheader("🚚 Supplier Contribution Analysis")

        supplier_count = st.slider(
            "Number of suppliers to display",
            min_value=5,
            max_value=30,
            value=15,
            step=1
        )

        supplier_summary = (
            filtered_df
            .groupby(
                "Supplier",
                as_index=False
            )
            .agg(
                Inventory_Value=(
                    "Inventory_Value",
                    "sum"
                ),
                Revenue_At_Risk=(
                    "Revenue_At_Risk",
                    "sum"
                ),
                Recovered_Revenue=(
                    "Recovered_Revenue",
                    "sum"
                ),
                Waste_Cost=(
                    "Waste_Cost",
                    "sum"
                ),
                Current_Stock=(
                    "Current_Stock",
                    "sum"
                )
            )
        )

        supplier_summary["Recovery_Rate_%"] = np.where(
            supplier_summary["Revenue_At_Risk"] > 0,
            (
                supplier_summary["Recovered_Revenue"]
                / supplier_summary["Revenue_At_Risk"]
            ) * 100,
            0
        )

        supplier_summary = (
            supplier_summary
            .sort_values(
                "Inventory_Value",
                ascending=False
            )
            .head(supplier_count)
        )

        fig = px.bar(
            supplier_summary,
            x="Supplier",
            y=[
                "Revenue_At_Risk",
                "Recovered_Revenue",
                "Waste_Cost"
            ],
            barmode="group",
            title=(
                "Supplier Revenue at Risk, "
                "Recovery and Waste Cost"
            ),
            labels={
                "value": "Amount (₹)",
                "variable": "Financial Metric"
            }
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.scatter(
            supplier_summary,
            x="Inventory_Value",
            y="Recovery_Rate_%",
            size="Current_Stock",
            color="Supplier",
            hover_name="Supplier",
            title=(
                "Supplier Inventory Value vs Recovery Rate"
            ),
            labels={
                "Inventory_Value": "Inventory Value (₹)",
                "Recovery_Rate_%": "Recovery Rate (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            supplier_summary,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 5. HISTORICAL TRENDS
    # =====================================================

    elif selected_chart == (
        "Historical Trends (Revenue & Inventory)"
    ):

        st.subheader(
            "📈 Historical Trends: Revenue and Inventory"
        )

        if (
            "Timestamp" not in filtered_df.columns
            or filtered_df["Timestamp"].isna().all()
        ):

            st.warning(
                "Valid Timestamp data is not available."
            )

        else:

            filtered_df["Analysis_Date"] = (
                pd.to_datetime(
                    filtered_df["Timestamp"],
                    errors="coerce"
                )
                .dt.date
            )

            trend_summary = (
                filtered_df
                .dropna(subset=["Analysis_Date"])
                .groupby(
                    "Analysis_Date",
                    as_index=False
                )
                .agg(
                    Inventory_Value=(
                        "Inventory_Value",
                        "sum"
                    ),
                    Estimated_Daily_Revenue=(
                        "Estimated_Daily_Revenue",
                        "sum"
                    ),
                    Revenue_At_Risk=(
                        "Revenue_At_Risk",
                        "sum"
                    ),
                    Recovered_Revenue=(
                        "Recovered_Revenue",
                        "sum"
                    ),
                    Current_Stock=(
                        "Current_Stock",
                        "sum"
                    )
                )
                .sort_values("Analysis_Date")
            )

            metric_selection = st.multiselect(
                "Select trend metrics",
                options=[
                    "Inventory_Value",
                    "Estimated_Daily_Revenue",
                    "Revenue_At_Risk",
                    "Recovered_Revenue"
                ],
                default=[
                    "Inventory_Value",
                    "Revenue_At_Risk",
                    "Recovered_Revenue"
                ]
            )

            if metric_selection:

                fig = px.line(
                    trend_summary,
                    x="Analysis_Date",
                    y=metric_selection,
                    markers=True,
                    title=(
                        "Historical Financial and "
                        "Inventory Trends"
                    ),
                    labels={
                        "value": "Amount (₹)",
                        "variable": "Metric",
                        "Analysis_Date": "Date"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            fig = px.area(
                trend_summary,
                x="Analysis_Date",
                y="Current_Stock",
                title="Historical Current Stock Trend",
                labels={
                    "Current_Stock": "Current Stock",
                    "Analysis_Date": "Date"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                trend_summary,
                use_container_width=True,
                hide_index=True
            )

    # =====================================================
    # 6. EXPIRY TIMELINES AND CAPITAL AT RISK
    # =====================================================

    elif selected_chart == (
        "Expiry Timelines & Capital At Risk"
    ):

        st.subheader(
            "⏳ Expiry Timelines and Capital at Risk"
        )

        risk_order = [
            "High",
            "Medium",
            "Low"
        ]

        tab1, tab2, tab3 = st.tabs(
            [
                "Expiry Timeline",
                "Risk Distribution",
                "Capital at Risk"
            ]
        )

        with tab1:

            fig = px.histogram(
                filtered_df,
                x="Days_Remaining",
                color="Expiry_Risk",
                nbins=40,
                category_orders={
                    "Expiry_Risk": risk_order
                },
                title=(
                    "Product Expiry Timeline Distribution"
                ),
                labels={
                    "Days_Remaining":
                        "Days Remaining to Expiry",
                    "count": "Number of Records"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab2:

            expiry_summary = (
                filtered_df
                .groupby(
                    "Expiry_Risk",
                    as_index=False
                )
                .agg(
                    Product_Count=(
                        "Product_ID",
                        "count"
                    ),
                    Revenue_At_Risk=(
                        "Revenue_At_Risk",
                        "sum"
                    ),
                    Inventory_Value=(
                        "Inventory_Value",
                        "sum"
                    )
                )
            )

            fig = px.pie(
                expiry_summary,
                names="Expiry_Risk",
                values="Product_Count",
                hole=0.45,
                title="Expiry Risk Distribution",
                category_orders={
                    "Expiry_Risk": risk_order
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab3:

            capital_risk = (
                filtered_df
                .groupby(
                    "Category",
                    as_index=False
                )
                .agg(
                    Revenue_At_Risk=(
                        "Revenue_At_Risk",
                        "sum"
                    ),
                    Waste_Cost=(
                        "Waste_Cost",
                        "sum"
                    ),
                    Recovered_Revenue=(
                        "Recovered_Revenue",
                        "sum"
                    )
                )
                .sort_values(
                    "Revenue_At_Risk",
                    ascending=False
                )
            )

            fig = px.bar(
                capital_risk,
                x="Category",
                y=[
                    "Revenue_At_Risk",
                    "Waste_Cost",
                    "Recovered_Revenue"
                ],
                barmode="group",
                title=(
                    "Capital at Risk and Recovery "
                    "by Category"
                ),
                labels={
                    "value": "Amount (₹)",
                    "variable": "Risk Metric"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        urgent_expiry = (
            filtered_df[
                (
                    filtered_df["Days_Remaining"] <= 30
                )
                | (
                    filtered_df["Expiry_Risk"]
                    .astype(str)
                    .str.lower()
                    .eq("high")
                )
            ]
            .sort_values(
                "Revenue_At_Risk",
                ascending=False
            )
            [
                [
                    "Product_ID",
                    "Product_Name",
                    "Category",
                    "Brand",
                    "Days_Remaining",
                    "Expiry_Risk",
                    "Current_Stock",
                    "Revenue_At_Risk",
                    "Recommended_Action"
                ]
            ]
            .head(100)
        )

        st.markdown("#### Urgent Expiry Products")

        st.dataframe(
            urgent_expiry,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 7. SALES AND PERFORMANCE RANKINGS
    # =====================================================

    elif selected_chart == "Sales & Performance Rankings":

        st.subheader("🏆 Sales and Performance Rankings")

        ranking_metric = st.selectbox(
            "Select ranking metric",
            options=[
                "Daily_Sales",
                "Monthly_Demand",
                "Demand_Forecast",
                "Estimated_Daily_Revenue",
                "Estimated_Daily_Profit",
                "Customer_Rating",
                "Recovered_Revenue"
            ],
            format_func=lambda value: (
                value.replace("_", " ")
            )
        )

        ranking_count = st.slider(
            "Number of products",
            min_value=5,
            max_value=30,
            value=15,
            step=1
        )

        top_products = (
            filtered_df
            .nlargest(
                ranking_count,
                ranking_metric
            )
            [
                [
                    "Product_ID",
                    "Product_Name",
                    "Category",
                    "Brand",
                    ranking_metric,
                    "Current_Stock",
                    "Expiry_Risk",
                    "Recommended_Action"
                ]
            ]
            .sort_values(
                ranking_metric,
                ascending=True
            )
        )

        fig = px.bar(
            top_products,
            x=ranking_metric,
            y="Product_Name",
            orientation="h",
            color=ranking_metric,
            hover_data=[
                "Product_ID",
                "Category",
                "Brand",
                "Current_Stock",
                "Expiry_Risk",
                "Recommended_Action"
            ],
            title=(
                f"Top {ranking_count} Products by "
                f"{ranking_metric.replace('_', ' ')}"
            ),
            labels={
                ranking_metric:
                    ranking_metric.replace("_", " "),
                "Product_Name": "Product"
            }
        )

        fig.update_layout(
            height=max(
                500,
                ranking_count * 35
            ),
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            action_summary = (
                filtered_df["Recommended_Action"]
                .value_counts()
                .reset_index()
            )

            action_summary.columns = [
                "Recommended_Action",
                "Count"
            ]

            fig = px.pie(
                action_summary,
                names="Recommended_Action",
                values="Count",
                hole=0.4,
                title="Recommended Action Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            prediction_summary = (
                filtered_df["Prediction_Result"]
                .value_counts()
                .reset_index()
            )

            prediction_summary.columns = [
                "Prediction_Result",
                "Count"
            ]

            fig = px.pie(
                prediction_summary,
                names="Prediction_Result",
                values="Count",
                hole=0.4,
                title="Prediction Result Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            top_products.sort_values(
                ranking_metric,
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # DOWNLOAD FILTERED EDA DATA
    # -----------------------------------------------------

    st.divider()

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered EDA Data",
        data=csv_data,
        file_name="smartinventory_eda_filtered_data.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# PAGE 7: Run AI Models
# -------------------------------------------------------------
elif page == "🤖 Run AI Models":
    st.title("AI Prediction Workspace")
    st.markdown("Use this page after Data Cleaning and Feature Engineering. Training creates reusable model files; prediction applies those saved models to the current dataset.")
    st.info("Normal use: click **Run Predictions on Current Dataset**. Use **Retrain Models** only when your dataset structure or historical patterns have materially changed.")
    
    if "engineered_df" not in st.session_state:
        st.warning("Please complete feature engineering on the 'Feature Engineering' page first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Retrain Models (optional)")
            st.write("Rebuild all four ML models using the current feature-engineered data.")
            st.markdown("- **Expiry Risk (Random Forest)**")
            st.markdown("- **Demand Forecast (XGBoost)**")
            st.markdown("- **Revenue Prediction (Linear Regression)**")
            st.markdown("- **Inventory Segmentation (K-Means Clustering)**")
            
            if st.button("Retrain Models", use_container_width=True):
                with st.spinner("Training machine learning models..."):
                    train_and_save_models(ENGINEERED_CSV_PATH, MODELS_DIR)
                    st.success("All models trained and saved successfully as .pkl files!")
                    
        with col2:
            st.subheader("2. Generate Predictions")
            st.write("Apply saved models to every product and calculate expiry risk, demand, revenue risk and inventory health.")
            
            if st.button("Run Predictions on Current Dataset", type="primary", use_container_width=True):
                if not os.path.exists(os.path.join(MODELS_DIR, "expiry_model.pkl")):
                    st.error("Model files not found! Please train the models first using the button on the left.")
                else:
                    with st.spinner("Executing model inference pipelines..."):
                        pred_df = run_predictions(CLEANED_CSV_PATH, MODELS_DIR, PREDICTION_CSV_PATH)
                        st.session_state["prediction_df"] = pred_df
                        st.success("Predictions completed. Open Prediction Results, then generate business recommendations.")
                        st.rerun()
                        
        # Display model status
        st.subheader("Model Storage Status")
        model_names = ["expiry_model.pkl", "demand_model.pkl", "revenue_model.pkl", "cluster_model.pkl"]
        for m_name in model_names:
            m_path = os.path.join(MODELS_DIR, m_name)
            if os.path.exists(m_path):
                st.write(f"✅ **{m_name}**: Available (Size: {os.path.getsize(m_path)/1024:.1f} KB)")
            else:
                st.write(f"❌ **{m_name}**: Not found. Training required.")

# -------------------------------------------------------------
# PAGE 8: Prediction Results
# -------------------------------------------------------------
elif page == "📈 Prediction Results":
    st.title("📈 Prediction Results Dashboard")
    st.markdown("Review predicted model outputs for Expiry Risks, Demand velocity, Revenue projection, and Performance cluster segments.")
    
    if "prediction_df" not in st.session_state:
        st.warning("Please run predictions on the 'Run AI Models' page first.")
    else:
        df = st.session_state["prediction_df"]
        
        st.subheader("Predicted Attributes Preview")
        st.dataframe(df[["Product_ID", "Product_Name", "Category", "Predicted_Expiry_Risk", "Predicted_Sales_Velocity", "Predicted_Revenue", "Predicted_Cluster", "Predicted_Revenue_at_Risk"]].head(10))
        
        st.subheader("Predicted Profiles Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Predicted_Expiry_Risk", color="Predicted_Expiry_Risk", title="Predicted Expiry Risk Distributions",
                               color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981", "No Risk": "#64748B"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df, x="Sales_Velocity", y="Predicted_Sales_Velocity", color="Category", title="Actual vs Predicted Sales Velocity Comparison",
                             labels={"Sales_Velocity": "Actual Velocity (daily)", "Predicted_Sales_Velocity": "Predicted Velocity (daily)"})
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# PAGE 9: AI Recommendations
# -------------------------------------------------------------
elif page == "💡 AI Recommendations":
    st.title("💡 Strategic Recommendations Engine")
    st.markdown("Decision support engine containing specific inventory adjustments, reorder quantities, and markdown actions.")
    
    if "prediction_df" not in st.session_state:
        st.warning("Please execute prediction inference on the 'Prediction' page first.")
    else:
        if st.button("Generate Strategic Recommendations"):
            with st.spinner("Formulating AI recommendations and calculating expected recovery margins..."):
                recs_df, summary = generate_recommendations(PREDICTION_CSV_PATH, RECOMMENDATION_CSV_PATH, SUMMARY_JSON_PATH)
                st.session_state["recs_df"] = recs_df
                st.session_state["business_summary"] = summary
                
                # Automatically save generated data into SQLite/MySQL database
                try:
                    engine = get_db_engine()
                    init_db(engine)
                    load_data_to_db(engine, ENGINEERED_CSV_PATH, PREDICTION_CSV_PATH, RECOMMENDATION_CSV_PATH)
                    st.success("Successfully synchronized and uploaded all pipelines to SQL Database!")
                except Exception as e:
                    st.warning(f"Database sync skipped/failed: {e}")
                    
                st.success("Recommendations and Business Impact reports generated!")
                st.rerun()
                
        if "recs_df" in st.session_state and "business_summary" in st.session_state:
            recs_df = st.session_state["recs_df"]
            summary = st.session_state["business_summary"]
            
            # KPI Cards (Business Impact Summary)
            st.subheader("Business Impact Summary")
            k_col1, k_col2, k_col3 = st.columns(3)
            k_col4, k_col5, k_col6 = st.columns(3)
            
            with k_col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Products Analysed</div>
                    <div class="kpi-val">{summary["Products_Analysed"]:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col2:
                st.markdown(f"""
                <div class="kpi-card risk">
                    <div class="kpi-label">High Risk Products</div>
                    <div class="kpi-val">{summary["High_Risk_Products"]:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col3:
                st.markdown(f"""
                <div class="kpi-card risk">
                    <div class="kpi-label">Revenue at Risk</div>
                    <div class="kpi-val">₹{summary["Revenue_at_Risk"]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col4:
                st.markdown(f"""
                <div class="kpi-card recovery">
                    <div class="kpi-label">Expected Recovery</div>
                    <div class="kpi-val">₹{summary["Expected_Recovery"]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col5:
                st.markdown(f"""
                <div class="kpi-card recovery">
                    <div class="kpi-label">Waste Reduction Rate</div>
                    <div class="kpi-val">{summary["Estimated_Waste_Reduction_Pct"]}%</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col6:
                st.markdown(f"""
                <div class="kpi-card health">
                    <div class="kpi-label">Overall Inventory Health</div>
                    <div class="kpi-val">{summary["Overall_Inventory_Health_Pct"]}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.metric(
                "Actual recovered revenue (reported in dataset)",
                f"₹{summary.get('Actual_Recovered_Revenue', 0):,.2f}",
                help="Historical recovered revenue supplied by the uploaded dataset; this is separate from the model's expected recovery.",
            )
                
            st.caption("Expected Recovery is calculated only from expiry-risk stock after the recommended markdown and is capped at Revenue at Risk. Reorder and transfer opportunities are not double-counted.")
            st.subheader("Actionable Recommendations Queue")
            
            # Dynamic Filter for Recommendations
            q1, q2, q3 = st.columns(3)
            rec_type_filter = q1.selectbox("Recommendation type", ["All"] + sorted(recs_df["Recommendation_Type"].dropna().unique().tolist()))
            priority_filter = q2.selectbox("Priority", ["All", "High", "Medium", "Low"])
            product_search = q3.text_input("Find product", placeholder="ID or product name")
            q4, q5 = st.columns(2)
            category_options = sorted(recs_df["Category"].dropna().astype(str).unique().tolist()) if "Category" in recs_df else []
            supplier_options = sorted(recs_df["Supplier"].dropna().astype(str).unique().tolist()) if "Supplier" in recs_df else []
            category_filter = q4.multiselect("Category", category_options)
            supplier_filter = q5.multiselect("Supplier", supplier_options)
            
            filtered_recs = recs_df.copy()
            if rec_type_filter != "All":
                filtered_recs = filtered_recs[filtered_recs["Recommendation_Type"] == rec_type_filter]
            if priority_filter != "All":
                filtered_recs = filtered_recs[filtered_recs["Priority_Level"] == priority_filter]
            if category_filter:
                filtered_recs = filtered_recs[filtered_recs["Category"].astype(str).isin(category_filter)]
            if supplier_filter:
                filtered_recs = filtered_recs[filtered_recs["Supplier"].astype(str).isin(supplier_filter)]
            if product_search.strip():
                needle = product_search.strip().lower()
                filtered_recs = filtered_recs[
                    filtered_recs["Product_ID"].astype(str).str.lower().str.contains(needle, regex=False)
                    | filtered_recs["Product_Name"].astype(str).str.lower().str.contains(needle, regex=False)
                ]
                
            st.dataframe(filtered_recs[["Product_ID", "Product_Name", "Recommendation_Type", "Action", "Expected_Revenue_Recovery", "Priority_Level", "Business_Reason"]], use_container_width=True)

# -------------------------------------------------------------
# PAGE 10: Executive Dashboard
# -------------------------------------------------------------
elif page == "🖥️ Executive Dashboard":
    st.title("Executive Inventory Command Center")
    st.markdown("A decision-focused view of stock exposure, demand, expiry risk and revenue protection opportunities.")
    
    if "prediction_df" not in st.session_state:
        st.warning("Please calculate model outputs and recommendations first.")
    else:
        pred_df = st.session_state["prediction_df"].copy()
        category_values = sorted(pred_df["Category"].dropna().astype(str).unique()) if "Category" in pred_df else []
        risk_values = sorted(pred_df["Predicted_Expiry_Risk"].dropna().astype(str).unique()) if "Predicted_Expiry_Risk" in pred_df else []
        f1, f2 = st.columns(2)
        selected_categories = f1.multiselect("Category", category_values, default=category_values)
        selected_risks = f2.multiselect("Expiry risk", risk_values, default=risk_values)
        if category_values:
            pred_df = pred_df[pred_df["Category"].astype(str).isin(selected_categories)]
        if risk_values:
            pred_df = pred_df[pred_df["Predicted_Expiry_Risk"].astype(str).isin(selected_risks)]
        if pred_df.empty:
            st.warning("No products match the selected filters.")
            st.stop()

        product_count = pred_df["Product_ID"].nunique()
        inventory_value = pd.to_numeric(pred_df.get("Inventory_Value", 0), errors="coerce").fillna(0).sum()
        revenue_risk = pd.to_numeric(pred_df.get("Predicted_Revenue_at_Risk", 0), errors="coerce").fillna(0).sum()
        avg_health = pd.to_numeric(pred_df.get("Predicted_Inventory_Health", 0), errors="coerce").fillna(0).mean()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Products in view", f"{product_count:,}")
        m2.metric("Inventory value", f"₹{inventory_value:,.0f}")
        m3.metric("Revenue at risk", f"₹{revenue_risk:,.0f}")
        m4.metric("Inventory health", f"{avg_health:.1f}%")
        
        # Setup Excel file builder
        if st.button("Generate & Compile Excel Dashboard Spreadsheet"):
            with st.spinner("Compiling styled Excel workbook..."):
                create_excel_dashboard(ENGINEERED_CSV_PATH, EXCEL_PATH)
                st.success("Excel Dashboard compiled successfully in dashboards/excel/!")
                
        # Decision visuals
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(pred_df, names="Predicted_Expiry_Risk", values="Predicted_Revenue_at_Risk",
                         hole=.58, title="Revenue Exposure by Expiry Risk",
                         color="Predicted_Expiry_Risk", color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981","No Risk":"#64748b"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            risk_summary = pred_df.groupby("Category")["Predicted_Revenue_at_Risk"].sum().reset_index()
            risk_summary = risk_summary.sort_values("Predicted_Revenue_at_Risk").tail(12)
            fig = px.bar(risk_summary, x="Predicted_Revenue_at_Risk", y="Category", orientation="h",
                         title="Highest Revenue-at-Risk Categories",
                         labels={"Predicted_Revenue_at_Risk":"Revenue at risk (₹)"}, color="Predicted_Revenue_at_Risk", color_continuous_scale="OrRd")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.scatter(pred_df, x="Predicted_Sales_Velocity", y="Stock_Quantity", size="Inventory_Value",
                             color="Predicted_Expiry_Risk", hover_name="Product_Name", title="Stock vs Forecast Demand",
                             labels={"Predicted_Sales_Velocity":"Forecast daily demand","Stock_Quantity":"Units in stock"},
                             color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981","No Risk":"#64748b"})
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            tree = pred_df.groupby(["Category","Predicted_Expiry_Risk"], as_index=False)["Inventory_Value"].sum()
            fig = px.treemap(tree, path=["Category","Predicted_Expiry_Risk"], values="Inventory_Value",
                             color="Inventory_Value", color_continuous_scale="Emrld", title="Inventory Capital Allocation")
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(avg_health), number={"suffix":"%"},
                title={"text":"Overall Inventory Health"}, gauge={"axis":{"range":[0,100]},
                "bar":{"color":"#10b981"}, "steps":[{"range":[0,50],"color":"#7f1d1d"},{"range":[50,75],"color":"#78350f"},{"range":[75,100],"color":"#064e3b"}]}))
            gauge.update_layout(height=360)
            st.plotly_chart(gauge, use_container_width=True)
        with col6:
            heat = pd.crosstab(pred_df["Category"], pred_df["Predicted_Expiry_Risk"])
            heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).head(12).index]
            fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
                            title="Product Count Risk Heatmap", labels={"x":"Expiry risk","y":"Category","color":"Products"})
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Revenue at risk = selling value of stock classified as High or Medium expiry risk. Inventory health is a 0–100 composite from stock status, expiry risk and turnover.")

# -------------------------------------------------------------
# PAGE 11: Download Reports
# -------------------------------------------------------------
elif page == "📥 Download Reports":
    st.title("📥 Download Strategic Reports")
    st.markdown("Export cleaned datasets, prediction outputs, recommendation sheets, and executive spreadsheets.")
    
    # 1. Cleaned CSV
    if os.path.exists(CLEANED_CSV_PATH):
        with open(CLEANED_CSV_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Cleaned Dataset (.csv)",
                data=f,
                file_name="SmartInventory_AI_Cleaned_Dataset.csv",
                mime="text/csv"
            )
            
    # 2. Engineered CSV
    if os.path.exists(ENGINEERED_CSV_PATH):
        with open(ENGINEERED_CSV_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Feature Engineered Dataset (.csv)",
                data=f,
                file_name="SmartInventory_AI_Feature_Engineered.csv",
                mime="text/csv"
            )
            
    # 3. Predictions CSV
    if os.path.exists(PREDICTION_CSV_PATH):
        with open(PREDICTION_CSV_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Prediction Report (.csv)",
                data=f,
                file_name="Prediction_Report.csv",
                mime="text/csv"
            )
            
    # 4. Recommendation CSV
    if os.path.exists(RECOMMENDATION_CSV_PATH):
        with open(RECOMMENDATION_CSV_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Recommendation Decisions Report (.csv)",
                data=f,
                file_name="Recommendation_Report.csv",
                mime="text/csv"
            )
            
    # 5. Business Summary JSON
    if os.path.exists(SUMMARY_JSON_PATH):
        with open(SUMMARY_JSON_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Business Impact Summary (.json)",
                data=f,
                file_name="Business_Impact_Summary.json",
                mime="application/json"
            )
            
    # 6. Excel Dashboard Spreadsheet
    if os.path.exists(EXCEL_PATH):
        with open(EXCEL_PATH, "rb") as f:
            st.download_button(
                label="📊 Download Executive Excel Dashboard (.xlsx)",
                data=f,
                file_name="SmartInventory_AI_Executive_Dashboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Excel Dashboard has not been compiled yet. Please go to the 'Executive Dashboard' page and compile the Excel dashboard to download it here.")
        
    st.info("Power BI specification blueprints are saved inside dashboards/powerbi/powerbi_specs.md for reference.")

# -------------------------------------------------------------
# PAGE 12: AI BUSINESS ASSISTANT
# -------------------------------------------------------------
elif page == "💬 AI Business Assistant":
    st.title("AI Business Assistant")
    st.markdown(
        "Ask about the uploaded dataset, expiry exposure, demand, revenue, suppliers, "
        "recommendations or general inventory strategy. Company-specific answers are "
        "grounded only in computed pipeline results."
    )

    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    c1, c2 = st.columns([1, 4])
    if c1.button("Clear conversation"):
        st.session_state["ai_chat_history"] = []
        st.rerun()
    if os.getenv("GEMINI_API_KEY", "").strip():
        c2.success(f"AI connected from .env · {os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite')}")
    else:
        c2.warning("Add GEMINI_API_KEY in the project .env file, save it, and restart Streamlit.")

    for role, message in st.session_state["ai_chat_history"]:
        with st.chat_message(role):
            st.markdown(message)

    prompts = [
        "What are the top business priorities?",
        "Which products have the highest revenue risk?",
        "Explain the recommended actions and expected recovery.",
        "What information is missing from this dataset?",
    ]
    prompt_cols = st.columns(2)
    suggested = None
    for i, prompt_text in enumerate(prompts):
        if prompt_cols[i % 2].button(prompt_text, key=f"prompt_{i}", use_container_width=True):
            suggested = prompt_text

    question = st.chat_input("Ask SmartInventory AI...") or suggested
    if question:
        st.session_state["ai_chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing business context..."):
                response = ai_answer(
                    question,
                    st.session_state["ai_chat_history"],
                    raw_df=st.session_state.get("raw_df"),
                    prediction_df=st.session_state.get("prediction_df"),
                    recs_df=st.session_state.get("recs_df"),
                    summary=st.session_state.get("business_summary"),
                )
            st.markdown(response)
        st.session_state["ai_chat_history"].append(("assistant", response))
