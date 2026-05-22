from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_NO_PRECIP_VALS = {"NONE", "NO", "N", "NAN", ""}

_ENGINE_MAP = {
    "A": "Piston", "B": "Turbojet", "C": "Turboprop",
    "D": "Turbofan", "E": "Glider", "F": "Helicopter", "Y": "Other",
}

_DAMAGE_MAP = {
    "N": "None", "M": "Minor", "M?": "Undetermined",
    "S": "Substantial", "D": "Destroyed",
}

_PHASE_MAP = {
    "Take-off Run": "Takeoff", "Departure": "Takeoff", "Climb": "Climb",
    "En Route": "Cruise", "Descent": "Descent", "Approach": "Approach",
    "Arrival": "Approach", "Landing Roll": "Landing", "Taxi": "Ground",
    "Parked": "Ground", "Local": "Unknown", "Unknown": "Unknown",
}

_STATE_MAP = {
    "AL": "Alabama",    "AK": "Alaska",      "AZ": "Arizona",     "AR": "Arkansas",
    "CA": "California", "CO": "Colorado",    "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida",    "GA": "Georgia",     "HI": "Hawaii",      "ID": "Idaho",
    "IL": "Illinois",   "IN": "Indiana",     "IA": "Iowa",        "KS": "Kansas",
    "KY": "Kentucky",   "LA": "Louisiana",   "ME": "Maine",       "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",   "MS": "Mississippi",
    "MO": "Missouri",   "MT": "Montana",     "NE": "Nebraska",    "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",   "OK": "Oklahoma",
    "OR": "Oregon",     "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",       "UT": "Utah",
    "VT": "Vermont",    "VA": "Virginia",    "WA": "Washington",  "WV": "West Virginia",
    "WI": "Wisconsin",  "WY": "Wyoming",
}

_TERRITORY_MAP = {
    "PR": "Puerto Rico",      "VI": "U.S. Virgin Islands", "GU": "Guam",
    "AS": "American Samoa",   "MP": "Northern Mariana Islands",
    "UM": "U.S. Minor Outlying Islands",
}

_COLS_TO_KEEP = [
    "INCIDENT_DATE", "YEAR", "MONTH", "TIME", "TIME_OF_DAY", "AIRPORT",
    "STATE", "FAAREGION", "LATITUDE", "LONGITUDE", "RUNWAY", "OPID", "OPERATOR",
    "AIRCRAFT", "AC_CLASS", "AC_MASS", "NUM_ENGS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "SPECIES", "NUM_SEEN", "NUM_STRUCK", "SIZE", "DAMAGE_LEVEL", "NR_INJURIES",
    "NR_FATALITIES", "COST_REPAIRS", "COST_OTHER", "COST_REPAIRS_INFL_ADJ",
    "COST_OTHER_INFL_ADJ", "HEIGHT", "SPEED", "DISTANCE", "WARNED",
    "SKY", "PRECIP",
    "DAM_RAD", "DAM_WINDSHLD", "DAM_NOSE", "DAM_ENG1", "DAM_ENG2",
    "DAM_ENG3", "DAM_ENG4", "DAM_PROP", "DAM_WING_ROT", "DAM_FUSE",
    "DAM_LG", "DAM_TAIL", "DAM_LGHTS", "DAM_OTHER",
]

_DAM_COLS = [
    "DAM_RAD", "DAM_WINDSHLD", "DAM_NOSE", "DAM_ENG1", "DAM_ENG2",
    "DAM_ENG3", "DAM_ENG4", "DAM_PROP", "DAM_WING_ROT", "DAM_FUSE",
    "DAM_LG", "DAM_TAIL", "DAM_LGHTS", "DAM_OTHER",
]

_TEXT_COLS = [
    "STATE", "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPID", "OPERATOR",
    "AIRCRAFT", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT", "SPECIES",
    "NUM_SEEN", "NUM_STRUCK", "SIZE", "DAMAGE_LEVEL", "WARNED", "FAAREGION",
    "SKY", "PRECIP",
]

_DISPLAY_TEXT_COLS = [
    "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPERATOR", "AIRCRAFT", "AC_CLASS",
    "SPECIES", "SIZE", "WARNED", "ENGINE_TYPE", "DAMAGE_CATEGORY",
    "PHASE_GROUP", "OPERATOR_TYPE", "TIME_BUCKET", "SKY", "PRECIP", "BIRD_SEASON",
]

_CATEGORICAL_COLS = [
    "SIZE", "WARNED", "ENGINE_TYPE", "DAMAGE_CATEGORY", "PHASE_GROUP",
    "OPERATOR_TYPE", "TIME_BUCKET", "FAAREGION", "AC_CLASS",
    "SKY", "PRECIP", "BIRD_SEASON",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_raw_data(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, dtype=str, engine="openpyxl")
    else:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    return df


def _parse_precip(val):
    if pd.isna(val):
        return np.nan
    parts = [p.strip().upper() for p in str(val).split(",")]
    return 0.0 if all(p in _NO_PRECIP_VALS for p in parts) else 1.0


def _map_bird_season(month):
    month = pd.to_numeric(month, errors="coerce")
    if pd.isna(month):       return "Unknown"
    if month in [3, 4, 5]:  return "Spring"
    if month in [6, 7]:     return "Summer"
    if month in [8, 9, 10]: return "Fall"
    return "Winter"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _rename_columns(df):
    return df.rename(columns={
        "INCIDENT_YEAR":  "YEAR",
        "INCIDENT_MONTH": "MONTH",
        "PRECIPITATION":  "PRECIP",
        "BIRDS_SEEN":     "NUM_SEEN",   # legacy name guard
        "BIRDS_STRUCK":   "NUM_STRUCK", # legacy name guard
    })


def _apply_filters(df):
    return df[
        df["FAAREGION"].notna() &
        (df["FAAREGION"].str.strip().str.upper() != "FGN")
    ].copy()


def _select_base_columns(df):
    existing = [c for c in _COLS_TO_KEEP if c in df.columns]
    df = df[existing].copy()
    for col in _TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").str.strip()
    return df


def _convert_numerics(df):
    numeric_cols = [
        "LATITUDE", "LONGITUDE", "AC_MASS", "HEIGHT", "SPEED", "DISTANCE",
        "NR_INJURIES", "NR_FATALITIES", "COST_REPAIRS", "COST_OTHER",
        "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
    ]
    present = [c for c in numeric_cols if c in df.columns]
    df[present] = df[present].apply(pd.to_numeric, errors="coerce")

    if "NUM_ENGS" in df.columns:
        df["NUM_ENGS"] = (
            pd.to_numeric(df["NUM_ENGS"], errors="coerce")
            .astype("Int64")
            .astype("string")
            .fillna("Unknown")
        )
    return df


def _engineer_datetime(df):
    df["INCIDENT_DATE"] = pd.to_datetime(
        df["INCIDENT_DATE"], format="%Y-%m-%d", errors="coerce"
    )
    df["QUARTER"] = df["INCIDENT_DATE"].dt.quarter

    time_dt = pd.to_datetime(df["TIME"], format="%H:%M", errors="coerce")
    df["HOUR"] = time_dt.dt.hour
    df["TIME"] = time_dt.dt.time

    df["TIME_BUCKET"] = pd.cut(
        df["HOUR"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )
    df["TIME_BUCKET"] = (
        df["TIME_BUCKET"].cat.add_categories("Unknown").fillna("Unknown").astype(str)
    )
    return df


def _engineer_targets_and_flags(df):
    df["TOTAL_COST"] = df["COST_REPAIRS"].fillna(0) + df["COST_OTHER"].fillna(0)
    df["TOTAL_COST_INFL_ADJ"] = (
        df["COST_REPAIRS_INFL_ADJ"].fillna(0) + df["COST_OTHER_INFL_ADJ"].fillna(0)
    )
    df["HAS_DAMAGE"]  = np.where(df["TOTAL_COST"] > 0, 1, 0)
    df["HAS_INJURY"]  = (df["NR_INJURIES"] > 0).astype("Int8")
    df["HAS_FATALITY"] = (df["NR_FATALITIES"] > 0).astype("Int8")
    df["WARNED_FLAG"] = np.where(
        df["WARNED"] == "Yes", 1, np.where(df["WARNED"] == "No", 0, np.nan)
    )

    if "PRECIP" in df.columns:
        df["HAS_PRECIP"] = df["PRECIP"].apply(_parse_precip)

    if "NUM_STRUCK" in df.columns:
        struck_upper = df["NUM_STRUCK"].astype(str).str.upper()
        df["IS_MULTI_STRIKE"] = (
            struck_upper.isin(["2-10", "11-100"]) |
            struck_upper.str.contains("MORE", na=False)
        ).astype(int)

    if "MONTH" in df.columns:
        df["BIRD_SEASON"] = df["MONTH"].apply(_map_bird_season)

    return df


def _apply_categorical_mappings(df):
    if "TYPE_ENG" in df.columns:
        df["ENGINE_TYPE"] = df["TYPE_ENG"].map(_ENGINE_MAP).fillna("Unknown")

    if "DAMAGE_LEVEL" in df.columns:
        df["DAMAGE_CATEGORY"] = df["DAMAGE_LEVEL"].map(_DAMAGE_MAP)
        df.loc[df["DAMAGE_CATEGORY"].isna() & (df["HAS_DAMAGE"] == 1), "DAMAGE_CATEGORY"] = "Undetermined"
        df["DAMAGE_CATEGORY"] = df["DAMAGE_CATEGORY"].fillna("None")

    if "PHASE_OF_FLIGHT" in df.columns:
        df["PHASE_GROUP"] = df["PHASE_OF_FLIGHT"].map(_PHASE_MAP).fillna("Unknown")

    if "OPID" in df.columns:
        df["OPERATOR_TYPE"] = df["OPID"].replace({
            "PVT": "Private", "BUS": "Business",
            "GOV": "Government", "MIL": "Military",
        }).fillna("Unknown")

    return df


def _engineer_geography(df):
    full_region_map = {**_STATE_MAP, **_TERRITORY_MAP}
    df["REGION_CLEAN"] = df["STATE"].map(full_region_map).fillna("Unknown")
    df["IS_US_STATE"]  = df["STATE"].isin(_STATE_MAP).astype(int)
    return df


def _format_display(df):
    for col in _DISPLAY_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").str.strip().str.title()

    df["FAAREGION"] = df["FAAREGION"].astype(str).str.strip().str.upper()

    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def _cleanup(df):
    existing_dam_cols = [c for c in _DAM_COLS if c in df.columns]
    drop = ["TYPE_ENG", "DAMAGE_LEVEL", "PHASE_OF_FLIGHT"] + existing_dam_cols
    return df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    base_dir   = Path(__file__).resolve().parent.parent
    input_path = base_dir / "data" / "raw" / "Public.xlsx"
    output_dir = base_dir / "data" / "processed" / "app"
    parquet_path = output_dir / "app_data_cleaned.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Initializing Production App Dashboard Data Cleaning Pipeline...")
    df = load_raw_data(input_path)
    df = _rename_columns(df)
    df = _apply_filters(df)
    df = _select_base_columns(df)
    df = _convert_numerics(df)
    df = _engineer_datetime(df)
    df = _engineer_targets_and_flags(df)
    df = _apply_categorical_mappings(df)
    df = _engineer_geography(df)
    df = _format_display(df)
    df = _cleanup(df)

    df.to_parquet(parquet_path, compression="snappy", index=False)
    print(f"✅ Success! Dashboard dataset saved to:\n📦 {parquet_path}")


if __name__ == "__main__":
    main()