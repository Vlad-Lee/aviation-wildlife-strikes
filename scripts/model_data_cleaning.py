from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile


def main():
    # -------------------------------------------------------------------------
    # 1. Paths & Directory Architecture
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent.parent

    input_path = base_dir / "data" / "raw" / "raw_data.csv"
    output_dir = base_dir / "data" / "processed" / "modeling"

    # Ensure the exact nested modeling output folder structure exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Initializing Production Modeling Data Cleaning Pipeline...")
    df = pd.read_csv(input_path, low_memory=False)

    # Standardize column header casings to minimize lookup mismatches
    df.columns = df.columns.str.strip().str.upper()

    # Rename temporal features for readability
    df = df.rename(columns={"INCIDENT_YEAR": "YEAR", "INCIDENT_MONTH": "MONTH"})


    # -------------------------------------------------------------------------
    # 2. Vectorized Feature Calculations & Text Formatting
    # -------------------------------------------------------------------------
    # Safe categorical text normalization across raw features
    text_cols = [
        "STATE", "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPID", "OPERATOR",
        "AIRCRAFT", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT", "SPECIES",
        "NUM_SEEN", "NUM_STRUCK", "SIZE", "DAMAGE_LEVEL", "WARNED", "AC_MASS"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").str.strip()

    # Strict numeric element conversion
    numeric_cols = [
        "LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE",
        "NR_INJURIES", "NR_FATALITIES", "COST_REPAIRS", "COST_OTHER",
        "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Standardize number of engines as a categorical feature
    df["NUM_ENGS"] = (
        pd.to_numeric(df["NUM_ENGS"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .fillna("Unknown")
    )


    # -------------------------------------------------------------------------
    # 3. Target Variable & Flag Engineering
    # -------------------------------------------------------------------------
    # Stage 1 target conditions
    damage_level = df["DAMAGE_LEVEL"].isin(["M", "S", "D", "M?"])
    
    # Indicator columns for part-specific damage
    dam_cols = [
        'DAM_RAD', 'DAM_WINDSHLD', 'DAM_NOSE', 'DAM_ENG1', 'DAM_ENG2', 
        'DAM_ENG3', 'DAM_ENG4', 'DAM_PROP', 'DAM_WING_ROT', 'DAM_FUSE', 
        'DAM_LG', 'DAM_TAIL', 'DAM_LGHTS', 'DAM_OTHER'
    ]

    # Check if ANY of the damage columns are flagged as 1 for that row
    existing_dam_cols = [col for col in dam_cols if col in df.columns]
    cond_component_damage = (df[existing_dam_cols] == 1).any(axis=1)

    # Establish the unified binary classification target
    df["HAS_DAMAGE"] = np.where(
        damage_level | cond_component_damage,
        1,
        np.where(df["DAMAGE_LEVEL"] == "N", 0, np.nan)
    )

    # Stage 2 Target Condition: Total Inflation Adjusted Financial Harm
    cost_rep = df["COST_REPAIRS_INFL_ADJ"].fillna(0)
    cost_oth = df["COST_OTHER_INFL_ADJ"].fillna(0)
    df["TOTAL_COST_INFL_ADJ"] = cost_rep + cost_oth

    # Operational Warning Flags
    df["WARNED_FLAG"] = np.where(
        df["WARNED"] == "Yes", 1, np.where(df["WARNED"] == "No", 0, 0)
    )


    # -------------------------------------------------------------------------
    # 4. Date and Time Feature Engineering
    # -------------------------------------------------------------------------
    df["INCIDENT_DATE"] = pd.to_datetime(
        df["INCIDENT_DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )
    df["QUARTER"] = df["INCIDENT_DATE"].dt.quarter

    # Parse raw time strings to extract hours
    time_dt = pd.to_datetime(df["TIME"], format="%H:%M", errors="coerce")
    df["HOUR"] = time_dt.dt.hour

    # Bin hours into your categorical TIME_BUCKET
    df["TIME_BUCKET"] = pd.cut(
        df["HOUR"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )
    
    # Convert the Categorical type to string and handle nulls safely
    df["TIME_BUCKET"] = df["TIME_BUCKET"].astype(str).fillna("UNKNOWN").str.upper()


    # -------------------------------------------------------------------------
    # 5. Categorical Mappings
    # -------------------------------------------------------------------------
    engine_map = {
        "A": "Piston", "B": "Turbojet", "C": "Turboprop", "D": "Turbofan",
        "E": "Glider", "F": "Helicopter", "Y": "Other", "Unknown": "Unknown"
    }
    df["ENGINE_TYPE"] = df["TYPE_ENG"].map(engine_map).fillna("Unknown")

    phase_map = {
        "Take-off Run": "Takeoff", "Departure": "Takeoff", "Climb": "Climb",
        "En Route": "Cruise", "Descent": "Descent", "Approach": "Approach",
        "Arrival": "Approach", "Landing Roll": "Landing", "Taxi": "Ground",
        "Parked": "Ground", "Local": "Unknown", "Unknown": "Unknown"
    }
    df["PHASE_GROUP"] = df["PHASE_OF_FLIGHT"].map(phase_map).fillna("Unknown")

    df["OPERATOR_TYPE"] = df["OPID"].replace({
        "PVT": "Private", "BUS": "Business", "GOV": "Government", "MIL": "Military"
    }).fillna("Unknown")

    # Final string element standardization to enforce clean groupings
    for col in ["SIZE", "AC_MASS", "ENGINE_TYPE", "PHASE_GROUP", "OPERATOR_TYPE", "FAAREGION", "TIME_BUCKET"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # -------------------------------------------------------------------------
    # 6. The Unified Data Funnel (Row Dropping Execution)
    # -------------------------------------------------------------------------
    print(f"📋 Initial raw lines logged: {len(df)}")

    # Isolate valid US geographic operational entries and known target records
    valid_geo_mask = (df["FAAREGION"] != "FGN") & (df["FAAREGION"] != "UNKNOWN") & (df["FAAREGION"].notna())
    known_target_mask = df["HAS_DAMAGE"].notna()

    # Apply unified matrix filter down to structural modeling scope
    df_model = df[valid_geo_mask & known_target_mask].copy()
    df_model["HAS_DAMAGE"] = df_model["HAS_DAMAGE"].astype(int)

    print(f"📊 Filtering complete. Safe modeling lines remaining: {len(df_model)}")


    # -------------------------------------------------------------------------
    # 7. Matrix Partitioning & Feature/Target Slicing
    # -------------------------------------------------------------------------
    features_to_keep = [
        "MONTH", "SPEED", "HEIGHT", "LATITUDE", "LONGITUDE", 
        "SIZE", "AC_MASS", "ENGINE_TYPE", "NUM_ENGS", "PHASE_GROUP", 
        "OPERATOR_TYPE", "FAAREGION", "WARNED_FLAG", "TIME_BUCKET"
    ]

    X = df_model[features_to_keep].copy()
    y_class = df_model["HAS_DAMAGE"]
    y_reg = df_model["TOTAL_COST_INFL_ADJ"]

    # Execute strict Stratified 80/20 train/test partition to lock down target split balance
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, 
        test_size=0.20, 
        random_state=13, 
        stratify=y_class
    )


    # -------------------------------------------------------------------------
    # 8. Isolated Continuous Factor Imputation (Zero Leakage Guardrail)
    # -------------------------------------------------------------------------
    # Derive positioning and velocity medians SOLELY based on training boundaries
    median_speed = X_train["SPEED"].median()
    median_height = X_train["HEIGHT"].median()
    median_lat = X_train["LATITUDE"].median()
    median_lon = X_train["LONGITUDE"].median()

    # Apply training baselines to Training block
    X_train["SPEED"] = X_train["SPEED"].fillna(median_speed)
    X_train["HEIGHT"] = X_train["HEIGHT"].fillna(median_height)
    X_train["LATITUDE"] = X_train["LATITUDE"].fillna(median_lat)
    X_train["LONGITUDE"] = X_train["LONGITUDE"].fillna(median_lon)

    # Broadcast training baselines directly to Testing block
    X_test["SPEED"] = X_test["SPEED"].fillna(median_speed)
    X_test["HEIGHT"] = X_test["HEIGHT"].fillna(median_height)
    X_test["LATITUDE"] = X_test["LATITUDE"].fillna(median_lat)
    X_test["LONGITUDE"] = X_test["LONGITUDE"].fillna(median_lon)

    # Catch remaining open text null markers safely
    X_train = X_train.fillna("UNKNOWN")
    X_test = X_test.fillna("UNKNOWN")


    # -------------------------------------------------------------------------
    # 9. Isolate Stage 2 Subsets (Financial Severity Matrix)
    # -------------------------------------------------------------------------
    # Filter strictly for flights that incurred non-zero physical damages
    damaged_train_mask = (y_class_train == 1)
    X_train_reg = X_train[damaged_train_mask].copy()
    
    # Target compression: Map highly skewed real currency to symmetric log spaces
    y_train_reg_log = np.log1p(y_reg_train[damaged_train_mask])


    # -------------------------------------------------------------------------
    # 10. Save Artifacts & ZIP Bundling
    # -------------------------------------------------------------------------
    # Save Stage 1 Base Classification sets
    X_train.to_csv(output_dir / "X_train_stage1.csv", index=False)
    y_class_train.to_csv(output_dir / "y_train_stage1.csv", index=False)
    X_test.to_csv(output_dir / "X_test_unfiltered.csv", index=False)
    y_class_test.to_csv(output_dir / "y_test_stage1.csv", index=False)
    
    # Save Stage 2 Truncated Log-Continuous Regression sets
    X_train_reg.to_csv(output_dir / "X_train_stage2.csv", index=False)
    y_train_reg_log.to_csv(output_dir / "y_train_stage2_log.csv", index=False)
    y_reg_test.to_csv(output_dir / "y_test_stage2_raw.csv", index=False)

    print(f"✅ Modeling files compiled and written successfully to: {output_dir}")
    print(f"📈 Stage 1 Partition splits: {len(X_train)} training entries | {len(X_test)} verification entries.")
    print(f"📉 Stage 2 Positive damage subset sizing: {len(X_train_reg)} training entries.")

    # Automatically compress all 7 modeling CSVs into a single clean deployment zip
    modeling_files = [
        "X_train_stage1.csv", "y_train_stage1.csv", "X_test_unfiltered.csv", "y_test_stage1.csv",
        "X_train_stage2.csv", "y_train_stage2_log.csv", "y_test_stage2_raw.csv"
    ]
    
    zip_model_path = output_dir / "modeling_matrices_package.zip"
    with zipfile.ZipFile(zip_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in modeling_files:
            file_to_zip = output_dir / filename
            if file_to_zip.exists():
                zipf.write(file_to_zip, arcname=filename)

    print(f"📦 Successfully compressed all 7 modeling matrices into:\n   {zip_model_path}")


if __name__ == "__main__":
    main()