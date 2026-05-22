import pandas as pd
import numpy as np
import zipfile
import io
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def clean_and_engineer(file_path):
    print("🚀 Loading and renaming raw data...")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()

    # Mirror app_data_cleaning.py filters for consistency
    df = df[df["FAAREGION"].notna() & (df["FAAREGION"].str.strip().str.upper() != "FGN")].copy()

    us_states = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
        "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
        "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
        "TX","UT","VT","VA","WA","WV","WI","WY"
    }
    df = df[df["STATE"].isin(us_states)].copy()

    # 1. IMMEDIATE RENAME & MAP: Force raw columns to analytical schema
    df = df.rename(columns={
        "TYPE_ENG": "ENGINE_TYPE"
    })
    
    # Standardize OPERATOR codes to 4 main categories (MIL, GOV, PVT, BUS)
    # to prevent feature explosion during one-hot encoding.
    def map_operator_to_category(name):
        name = str(name).upper()
        if any(word in name for word in ['MILITARY', 'NAVY', 'AIR FORCE', 'ARMY', 'USAF', 'USCG', 'COAST GUARD']):
            return 'MIL'
        if any(word in name for word in ['GOV', 'FED', 'STATE', 'COUNTY', 'CITY']):
            return 'GOV'
        if any(word in name for word in ['PRIVATE', 'PRIVATELY', 'PERSONAL', 'PVT', 'CHARTER']):
            return 'PVT'
        return 'BUS'
    
    # Apply mapping if the raw column exists
    if "OPERATOR" in df.columns:
        df["OPERATOR_CAT"] = df["OPERATOR"].apply(map_operator_to_category)

    phase_map = {
        "TAKE-OFF RUN": "TAKEOFF", "DEPARTURE": "TAKEOFF", "CLIMB": "CLIMB",
        "EN ROUTE": "CRUISE", "DESCENT": "DESCENT", "APPROACH": "APPROACH",
        "ARRIVAL": "APPROACH", "LANDING ROLL": "LANDING", "TAXI": "GROUND",
        "PARKED": "GROUND", "LOCAL": "UNKNOWN"
    }
    # Safely map phase, if column exists
    if "PHASE_OF_FLIGHT" in df.columns:
        df["PHASE_GROUP"] = df["PHASE_OF_FLIGHT"].map(phase_map).fillna("UNKNOWN")

        # Convert continuous height into meaningful altitude bins to handle extreme 
    # skew (high frequency of ground strikes) and improve feature robustness.
    def bin_height(height):
        height = pd.to_numeric(height, errors='coerce')
        if pd.isna(height) or height < 0: return "UNKNOWN"
        if height == 0: return "GROUND"
        if height < 500: return "LOW"
        if height < 3000: return "APPROACH_CLIMB"
        return "CRUISE"

    # Apply this in clean_and_engineer:
    df["HEIGHT_BIN"] = df["HEIGHT"].apply(bin_height)

    # 2. STRICT FILTERING: Prevent dummy variable explosion
    # UPDATED: Dropped "OPERATOR", Added "OPERATOR_CAT"
    features_to_keep = [
        "MONTH", "SPEED", "HEIGHT_BIN", "LATITUDE", "LONGITUDE", 
        "SIZE", "AC_MASS", "ENGINE_TYPE", "NUM_ENGS", "PHASE_GROUP", 
        "OPERATOR_CAT", "FAAREGION", "WARNED", "COST_REPAIRS_INFL_ADJ", 
        "COST_OTHER_INFL_ADJ"
    ]
    
    # Keep only columns that exist to prevent KeyErrors
    actual_cols = [c for c in features_to_keep if c in df.columns]
    df = df[actual_cols]

    print("⚙️ Engineering physics and target features...")
    # 3. PHYSICS ENGINEERING: Kinetic Energy Proxy
    size_mass_map = {"SMALL": 1, "MEDIUM": 5, "LARGE": 15}
    if "SIZE" in df.columns and "SPEED" in df.columns:
        df["SIZE_CLEAN"] = df["SIZE"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        df["KINETIC_ENERGY_PROXY"] = df["SIZE_CLEAN"].map(size_mass_map).fillna(1) * (pd.to_numeric(df["SPEED"], errors='coerce').fillna(0) ** 2)
        
    # Log transform the proxy
    df["LOG_KINETIC_ENERGY"] = np.log1p(df["KINETIC_ENERGY_PROXY"])

    # 4. TARGET ENGINEERING: The Hurdle
    df["TOTAL_COST"] = pd.to_numeric(df.get("COST_REPAIRS_INFL_ADJ", 0), errors='coerce').fillna(0) + \
                       pd.to_numeric(df.get("COST_OTHER_INFL_ADJ", 0), errors='coerce').fillna(0)
    
    df["HAS_DAMAGE"] = np.where(df["TOTAL_COST"] > 0, 1, 0)

    if "WARNED" in df.columns:
        df["WARNED_FLAG"] = np.where(df["WARNED"].astype(str).str.strip().str.upper() == "YES", 1, 0)

    # Clean up redundant columns
    cols_to_drop = ["COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "WARNED", "SIZE", "SIZE_CLEAN", "KINETIC_ENERGY_PROXY"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df

def split_and_impute(df):
    print("✂️ Splitting data to prevent information leakage...")
    target_stage1 = df["HAS_DAMAGE"]
    target_stage2 = df["TOTAL_COST"]
    features = df.drop(columns=["HAS_DAMAGE", "TOTAL_COST"])

    # 20% Holdout Vault
    X_train, X_test, y_train_stage1, y_test_stage1, y_train_stage2, y_test_stage2 = train_test_split(
        features, target_stage1, target_stage2, test_size=0.20, random_state=42, stratify=target_stage1
    )

    print("🩹 Imputing 'UNKNOWN' categories and missing values...")
    # Categorical Imputation: Replace "UNKNOWN" with the mode of the training set
    # UPDATED: Dropped "OPERATOR", Added "OPERATOR_CAT"
    cat_cols = ["AC_MASS", "HEIGHT_BIN", "PHASE_GROUP", "OPERATOR_CAT", "ENGINE_TYPE", "FAAREGION"]
    
    for col in cat_cols:
        if col in X_train.columns:
            # Standardize missing as "UNKNOWN"
            X_train[col] = X_train[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
            X_test[col] = X_test[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
            
            # Find the most frequent class (excluding "UNKNOWN" itself)
            valid_train = X_train[X_train[col] != "UNKNOWN"]
            if not valid_train.empty:
                train_mode = valid_train[col].mode().iloc[0] # Safely get the first mode value
                X_train[col] = X_train[col].replace("UNKNOWN", train_mode)
                X_test[col] = X_test[col].replace("UNKNOWN", train_mode)

    # Numerical Imputation: Median
    num_cols = ["SPEED", "LATITUDE", "LONGITUDE", "NUM_ENGS", "MONTH"]
    for col in num_cols:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            
            train_median = X_train[col].median()
            X_train[col] = X_train[col].fillna(train_median)
            X_test[col] = X_test[col].fillna(train_median)

    print("🔀 One-Hot Encoding categorical variables...")
    existing_cat_cols = [c for c in cat_cols if c in X_train.columns]
    X_train_encoded = pd.get_dummies(X_train, columns=existing_cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=existing_cat_cols, drop_first=True)
    
    # Ensure train and test have the exact same columns after dummy encoding
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    # Stage 2 specific matrices (Only where a cost occurred)
    train_mask_stage2 = y_train_stage2 > 0
    X_train_stage2 = X_train_encoded[train_mask_stage2]
    y_train_stage2_log = np.log1p(y_train_stage2[train_mask_stage2])

    return X_train_encoded, X_test_encoded, y_train_stage1, y_test_stage1, X_train_stage2, y_train_stage2_log, y_test_stage2

def package_matrices(matrices, out_dir):
    print("📦 Packaging matrices into ZIP...")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "modeling_matrices_package.zip")

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in matrices.items():
            csv_buffer = io.StringIO()
            # Convert series to dataframe so they package cleanly
            if isinstance(df, pd.Series):
                df = df.to_frame(name)
            df.to_csv(csv_buffer, index=False)
            z.writestr(f"{name}.csv", csv_buffer.getvalue())

    print(f"✅ Data cleaning complete! Modeling matrices saved to {zip_path}")

if __name__ == "__main__":
    # Resolve directories safely regardless of where the script is run from
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "raw_data.csv"
    out_dir = base_dir / "data" / "processed" / "modeling"

    # Run the pipeline
    df_clean = clean_and_engineer(data_path)
    X_train_s1, X_test_unf, y_train_s1, y_test_s1, X_train_s2, y_train_s2_log, y_test_s2_raw = split_and_impute(df_clean)

    matrices = {
        "X_train_stage1": X_train_s1,
        "X_test_unfiltered": X_test_unf,
        "y_train_stage1": y_train_s1,
        "y_test_stage1": y_test_s1,
        "X_train_stage2": X_train_s2,
        "y_train_stage2_log": y_train_s2_log,
        "y_test_stage2_raw": y_test_s2_raw
    }

    package_matrices(matrices, out_dir)