import io
import zipfile

import pandas as pd
import requests
import streamlit as st

@st.cache_data
def load_data():
    """Load cleaned wildlife strike data from a zipped parquet file."""
    url = "https://github.com/user-attachments/files/27189129/cleaned_wildlife_strikes.zip"

    r = requests.get(url)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    with z.open("cleaned_wildlife_strikes.parquet") as f:
        return pd.read_parquet(f)