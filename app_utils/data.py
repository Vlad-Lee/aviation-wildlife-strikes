import io
import zipfile
from pathlib import Path
import pandas as pd
import streamlit as st

# Dynamically point to the root of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    """Load cleaned wildlife strike data from a zipped parquet file locally."""
    # Point to the local processed data zip file
    data_path = ROOT_DIR / "data" / "processed" / "app" / "app_data_package.zip"
    
    if not data_path.exists():
        st.error(f"Data file not found at {data_path}. Please run your data prep script.")
        return pd.DataFrame()

    with zipfile.ZipFile(data_path, "r") as z:
        with z.open("app_data_cleaned.parquet") as f:
            return pd.read_parquet(f)