from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    base_dir = Path(__file__).resolve().parent.parent

    input_path = base_dir / "data" / "raw" / "raw_data.csv"
    output_path = base_dir / "data" / "processed" / "cleaned_wildlife_strikes.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load and select columns
    # -----------------------------
    df = pd.read_csv(input_path)

    cols_to_keep = [
        "INCIDENT_DATE", "INCIDENT_YEAR", "INCIDENT_MONTH", "TIME", "TIME_OF_DAY",
        "AIRPORT", "STATE", "LATITUDE", "LONGITUDE", "RUNWAY", "OPID", "OPERATOR",
        "AIRCRAFT", "AC_CLASS", "AC_MASS", "NUM_ENGS", "TYPE_ENG",
        "PHASE_OF_FLIGHT", "SPECIES", "NUM_SEEN", "NUM_STRUCK", "SIZE",
        "DAMAGE_LEVEL", "NR_INJURIES", "NR_FATALITIES", "COST_REPAIRS",
        "COST_OTHER", "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
        "HEIGHT", "SPEED", "DISTANCE", "WARNED",
    ]

    df = df[cols_to_keep].copy()

    df = df.rename(
        columns={
            "INCIDENT_YEAR": "YEAR",
            "INCIDENT_MONTH": "MONTH",
        }
    )

    # -----------------------------
    # Basic text cleanup
    # -----------------------------
    text_cols = [
        "STATE", "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPID", "OPERATOR",
        "AIRCRAFT", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT", "SPECIES",
        "NUM_SEEN", "NUM_STRUCK", "SIZE", "DAMAGE_LEVEL", "WARNED",
    ]

    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("Unknown")
            .str.strip()
        )

    # -----------------------------
    # Numeric conversions
    # -----------------------------
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

    # -----------------------------
    # Date and time features
    # -----------------------------
    df["INCIDENT_DATE"] = pd.to_datetime(
        df["INCIDENT_DATE"],
        format="%Y-%m-%d",
        errors="coerce",
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

    # -----------------------------
    # Derived metrics and flags
    # -----------------------------
    df["TOTAL_COST"] = df["COST_REPAIRS"] + df["COST_OTHER"]

    df["TOTAL_COST_INFL_ADJ"] = (
        df["COST_REPAIRS_INFL_ADJ"] + df["COST_OTHER_INFL_ADJ"]
    )

    df["HAS_DAMAGE"] = np.where(
        df["DAMAGE_LEVEL"].isin(["M", "S", "D", "M?"]),
        1,
        np.where(df["DAMAGE_LEVEL"] == "N", 0, np.nan),
    )

    df["HAS_INJURY"] = np.where(
        df["NR_INJURIES"].isna(),
        np.nan,
        (df["NR_INJURIES"] > 0).astype(int),
    )

    df["HAS_FATALITY"] = np.where(
        df["NR_FATALITIES"].isna(),
        np.nan,
        (df["NR_FATALITIES"] > 0).astype(int),
    )

    df["WARNED_FLAG"] = np.where(
        df["WARNED"] == "Yes",
        1,
        np.where(df["WARNED"] == "No", 0, np.nan),
    )

    # -----------------------------
    # Category mappings
    # -----------------------------
    engine_map = {
        "A": "Piston",
        "B": "Turbojet",
        "C": "Turboprop",
        "D": "Turbofan",
        "E": "Glider",
        "F": "Helicopter",
        "Y": "Other",
        "Unknown": "Unknown",
    }

    df["ENGINE_TYPE"] = df["TYPE_ENG"].map(engine_map).fillna("Unknown")

    damage_map = {
        "N": "None",
        "M": "Minor",
        "M?": "Undetermined",
        "S": "Substantial",
        "D": "Destroyed",
        "Unknown": "Unknown",
    }

    df["DAMAGE_CATEGORY"] = df["DAMAGE_LEVEL"].map(damage_map).fillna("Unknown")

    phase_map = {
        "Take-off Run": "Takeoff",
        "Departure": "Takeoff",
        "Climb": "Climb",
        "En Route": "Cruise",
        "Descent": "Descent",
        "Approach": "Approach",
        "Arrival": "Approach",
        "Landing Roll": "Landing",
        "Taxi": "Ground",
        "Parked": "Ground",
        "Local": "Unknown",
        "Unknown": "Unknown",
    }

    df["PHASE_GROUP"] = df["PHASE_OF_FLIGHT"].map(phase_map).fillna("Unknown")

    df["OPERATOR_TYPE"] = df["OPID"].replace(
        {
            "PVT": "Private",
            "BUS": "Business",
            "GOV": "Government",
            "MIL": "Military",
        }
    )

    # -----------------------------
    # Geography flags
    # -----------------------------
    state_map = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
        "WI": "Wisconsin", "WY": "Wyoming",
    }

    territory_map = {
        "PR": "Puerto Rico",
        "VI": "U.S. Virgin Islands",
        "GU": "Guam",
        "AS": "American Samoa",
        "MP": "Northern Mariana Islands",
        "UM": "U.S. Minor Outlying Islands",
    }

    def map_region(state: str) -> str:
        if state in state_map:
            return state_map[state]
        if state in territory_map:
            return territory_map[state]
        if state == "Unknown":
            return "Unknown"
        return "International"

    df["REGION_CLEAN"] = df["STATE"].apply(map_region)
    df["IS_US_STATE"] = df["STATE"].isin(state_map).astype(int)

    # -----------------------------
    # Final display formatting
    # -----------------------------
    display_text_cols = [
        "TIME_OF_DAY", "AIRPORT", "RUNWAY", "OPERATOR", "AIRCRAFT",
        "AC_CLASS", "SPECIES", "SIZE", "WARNED", "ENGINE_TYPE",
        "DAMAGE_CATEGORY", "PHASE_GROUP", "OPERATOR_TYPE",
    ]

    for col in display_text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("Unknown")
            .str.strip()
            .str.title()
        )

    # -----------------------------
    # Drop unused columns
    # -----------------------------
    df = df.drop(
        columns=[
            "TYPE_ENG",
            "DAMAGE_LEVEL",
            "PHASE_OF_FLIGHT",
        ]
    )

    # -----------------------------
    # Save
    # -----------------------------
    df.to_parquet(
        output_path,
        compression="snappy",
        index=False,
    )

if __name__ == "__main__":
    main()

