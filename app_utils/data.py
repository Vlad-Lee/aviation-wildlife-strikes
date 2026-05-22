import pandas as pd
import streamlit as st
from pathlib import Path

# Dynamically point to the root of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    """Load cleaned wildlife strike data from parquet file."""
    # Point to the cleaned parquet file created by app_data_cleaning.py
    data_path = ROOT_DIR / "data" / "processed" / "app" / "app_data_cleaned.parquet"
    
    if not data_path.exists():
        st.error(f"❌ Data file not found at {data_path}")
        st.info("💡 Run this first to generate the data:\n`python app_data_cleaning.py`")
        return pd.DataFrame()

    return pd.read_parquet(data_path)