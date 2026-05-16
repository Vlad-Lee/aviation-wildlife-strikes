import io
import zipfile

import pandas as pd
import requests
import streamlit as st

@st.cache_data
def load_data():
    """Load cleaned wildlife strike data from a zipped parquet file."""
    # UPDATED: Points permanently to the raw file asset tracked in your main repository branch
    url = "https://github.com/Vlad-Lee/aviation-wildlife-strikes/raw/main/data/processed/app/app_data_package.zip"

    r = requests.get(url)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    with z.open("app_data_cleaned.parquet") as f:
        return pd.read_parquet(f)