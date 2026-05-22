from pathlib import Path
import numpy as np
import pandas as pd
import zipfile


def main():
    # -------------------------------------------------------------------------
    # 1. Paths & Directory Setup
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent.parent

    input_path = base_dir / "data" / "raw" / "raw_data.csv"
    output_dir = base_dir / "data" / "processed" / "app"
    
    parquet_path = output_dir / "app_data_cleaned.parquet"
    csv_path = output_dir / "app_data_cleaned.csv"
    zip_output_path = output_dir / "app_data_package.zip"

    # Ensure the exact nested app output folder structure exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Initializing Production App Dashboard Data Cleaning Pipeline...")
    df = pd.read_csv(input_path, low_memory=False)

    # Force standard uppercase column casings to minimize lookups mismatching
    df.columns = df.columns.str.strip().str.upper()

    # Rename chronological columns for clear layout readability
    df = df.rename(columns={"INCIDENT_YEAR": "YEAR", "INCIDENT_MONTH": "MONTH"})

    # -------------------------------------------------------------------------
    # 2. Strict Filter: Focus on Known US Airspace (Matches Modeling Pipeline)
    # -------------------------------------------------------------------------
    # Remove international edge cases and rows missing region descriptors entirely
    df = df[df["FAAREGION"].notna() & (df["FAAREGION"].str.strip().str.upper() != "FGN")].copy()

    # -------------------------------------------------------------------------
    # 3. Base Column Selection & Initial Text Processing
    # -------------------------------------------------------------------------
    cols_to_keep = [
        "INCIDENT_DATE", "YEAR", "MONTH", "TIME", "TIME_OF_DAY", "AIRPORT", 
        "STATE", "FAAREGION", "LATITUDE", "LONGITUDE", "RUNWAY", "OPID", "OPERATOR",
        "AIRCRAFT", "AC_CLASS", "AC_MASS", "NUM_ENGS", "TYPE_ENG", "PHASE_OF_FLIGHT", 
        "SPECIES", "NUM_SEEN", "NUM_STRUCK", "SIZE", "DAMAGE_LEVEL", "NR_INJURIES", 
        "NR_FATALITIES", "COST_REPAIRS", "COST_OTHER", "COST_REPAIRS_INFL_ADJ", 
        "COST_OTHER_INFL_ADJ", "HEIGHT", "SPEED", "DISTANCE", "WARNED",
        # Explicit component columns required for target synchronization
        'DAM_RAD', 'DAM_WINDSHLD', 'DAM_NOSE', 'DAM_ENG1', 'DAM_ENG2', 
        'DAM_ENG3', 'DAM_ENG4', 'DAM_PROP', 'DAM_WING_ROT', 'DAM_FUSE', 
        'DAM_LG', 'DAM_TAIL', 'DAM_LGHTS', 'DAM_OTHER'
    ]
    
    # Dynamically select only the matching subset columns present in raw files
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_cols].copy()

    # Initial safe string transformations
    text_cols = [
        "STATE", "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPID", "OPERATOR",
        "AIRCRAFT", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT", "SPECIES",
        "NUM_SEEN", "NUM_STRUCK", "SIZE", "DAMAGE_LEVEL", "WARNED", "FAAREGION"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").str.strip()

    # -------------------------------------------------------------------------
    # 4. Strict Numeric Conversions
    # -------------------------------------------------------------------------
    numeric_cols = [
        "LATITUDE", "LONGITUDE", "AC_MASS", "HEIGHT", "SPEED", "DISTANCE",
        "NR_INJURIES", "NR_FATALITIES", "COST_REPAIRS", "COST_OTHER",
        "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["NUM_ENGS"] = (
        pd.to_numeric(df["NUM_ENGS"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .fillna("Unknown")
    )

    # -------------------------------------------------------------------------
    # 5. Date and Time Features
    # -------------------------------------------------------------------------
    df["INCIDENT_DATE"] = pd.to_datetime(
        df["INCIDENT_DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )
    df["QUARTER"] = df["INCIDENT_DATE"].dt.quarter

    time_dt = pd.to_datetime(df["TIME"], format="%H:%M", errors="coerce")
    df["HOUR"] = time_dt.dt.hour
    df["TIME"] = time_dt.dt.time

    # Safely bin hours and handle missing times without breaking the categorical index
    df["TIME_BUCKET"] = pd.cut(
        df["HOUR"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )
    df["TIME_BUCKET"] = df["TIME_BUCKET"].cat.add_categories("Unknown").fillna("Unknown").astype(str)

    # -------------------------------------------------------------------------
    # 6. Synchronized Target Variable & Flag Engineering
    # -------------------------------------------------------------------------
    # Calculate costs
    df["TOTAL_COST"] = df["COST_REPAIRS"].fillna(0) + df["COST_OTHER"].fillna(0)
    df["TOTAL_COST_INFL_ADJ"] = df["COST_REPAIRS_INFL_ADJ"].fillna(0) + df["COST_OTHER_INFL_ADJ"].fillna(0)

    # Advanced Target Override Match: Summary field OR the 14 checkboxes
    cond_damage_level = df["DAMAGE_LEVEL"].isin(["M", "S", "D", "M?"])
    dam_cols = [
        'DAM_RAD', 'DAM_WINDSHLD', 'DAM_NOSE', 'DAM_ENG1', 'DAM_ENG2', 
        'DAM_ENG3', 'DAM_ENG4', 'DAM_PROP', 'DAM_WING_ROT', 'DAM_FUSE', 
        'DAM_LG', 'DAM_TAIL', 'DAM_LGHTS', 'DAM_OTHER'
    ]
    existing_dam_cols = [col for col in dam_cols if col in df.columns]
    cond_component_damage = (df[existing_dam_cols] == 1).any(axis=1)

    # Define damage strictly by the financial threshold (Total Cost > 0)
    df["HAS_DAMAGE"] = np.where(df["TOTAL_COST"] > 0, 1, 0)

    # Health and Safety indicators (Optimized boolean mapping)
    df["HAS_INJURY"] = (df["NR_INJURIES"] > 0).astype("Int8")
    df["HAS_FATALITY"] = (df["NR_FATALITIES"] > 0).astype("Int8")
    df["WARNED_FLAG"] = np.where(df["WARNED"] == "Yes", 1, np.where(df["WARNED"] == "No", 0, np.nan))

    # -------------------------------------------------------------------------
    # 7. Categorical Mapping Routines
    # -------------------------------------------------------------------------
    engine_map = {"A": "Piston", "B": "Turbojet", "C": "Turboprop", "D": "Turbofan", "E": "Glider", "F": "Helicopter", "Y": "Other"}
    df["ENGINE_TYPE"] = df["TYPE_ENG"].map(engine_map).fillna("Unknown")

    damage_map = {"N": "None", "M": "Minor", "M?": "Undetermined", "S": "Substantial", "D": "Destroyed"}
    # If explicit component damage exists but summary box was empty, label it cleanly as Undetermined
    df["DAMAGE_CATEGORY"] = df["DAMAGE_LEVEL"].map(damage_map)
    df.loc[df["DAMAGE_CATEGORY"].isna() & (df["HAS_DAMAGE"] == 1), "DAMAGE_CATEGORY"] = "Undetermined"
    df["DAMAGE_CATEGORY"] = df["DAMAGE_CATEGORY"].fillna("None")

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

    # -------------------------------------------------------------------------
    # 8. Vectorized Geography Flags
    # -------------------------------------------------------------------------
    state_map = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
        "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
        "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
        "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
        "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    }
    territory_map = {
        "PR": "Puerto Rico", "VI": "U.S. Virgin Islands", "GU": "Guam",
        "AS": "American Samoa", "MP": "Northern Mariana Islands", "UM": "U.S. Minor Outlying Islands",
    }

    # Vectorized state resolution for speed
    full_region_map = {**state_map, **territory_map}
    df["REGION_CLEAN"] = df["STATE"].map(full_region_map).fillna("Unknown")
    df["IS_US_STATE"] = df["STATE"].isin(state_map).astype(int)

    # -------------------------------------------------------------------------
    # 9. Final Display Formatting & Memory Optimization
    # -------------------------------------------------------------------------
    display_text_cols = [
        "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPERATOR", "AIRCRAFT",
        "AC_CLASS", "SPECIES", "SIZE", "WARNED", "ENGINE_TYPE",
        "DAMAGE_CATEGORY", "PHASE_GROUP", "OPERATOR_TYPE", "TIME_BUCKET"
    ]
    for col in display_text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").str.strip().str.title()

    # Standardize explicit administrative region characters
    df["FAAREGION"] = df["FAAREGION"].astype(str).str.strip().str.upper()

    # Convert low-cardinality strings to Categories to save massive amounts of RAM
    categorical_cols = [
        "SIZE", "WARNED", "ENGINE_TYPE", "DAMAGE_CATEGORY", 
        "PHASE_GROUP", "OPERATOR_TYPE", "TIME_BUCKET", "FAAREGION", "AC_CLASS"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # -------------------------------------------------------------------------
    # 10. Clean-Up and Drop Transformed/Intermediate Tracking Vectors
    # -------------------------------------------------------------------------
    columns_to_drop = ["TYPE_ENG", "DAMAGE_LEVEL", "PHASE_OF_FLIGHT"] + existing_dam_cols
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

    # -------------------------------------------------------------------------
    # 11. Serialization
    # -------------------------------------------------------------------------
    df.to_parquet(parquet_path, compression="snappy", index=False)
    df.to_csv(csv_path, index=False)

    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # ARCNAME maps the file inside the zip archive cleanly without deep nested local paths
        zipf.write(parquet_path, arcname=parquet_path.name)
        zipf.write(csv_path, arcname=csv_path.name)
        
    print(f"✅ Success! Dashboard dataset saved to:\n📦 {parquet_path}\n📝 {csv_path}\n🗜️ {zip_output_path}")


if __name__ == "__main__":
    main()