from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats

###################################################################################################
#Calculate summary metrics.
def get_summary_metrics(df: pd.DataFrame, df_states: pd.DataFrame) -> dict:
    total_rows = len(df)
    
    return {
        "total_bird_strikes": total_rows,
        "total_damaged_aircraft": int(df["HAS_DAMAGE"].sum()),
        "damage_rate_pct": round((df["HAS_DAMAGE"].sum() / total_rows) * 100, 2) if total_rows > 0 else 0,
        "total_cost": df["TOTAL_COST"].sum(),
        "total_cost_infl_adj": df["TOTAL_COST_INFL_ADJ"].sum(),
        "median_cost_infl_adj": float(df[df["TOTAL_COST_INFL_ADJ"] > 0]["TOTAL_COST_INFL_ADJ"].median()),
        "total_fatalities": df["NR_FATALITIES"].sum(),
        "total_injuries": df["NR_INJURIES"].sum(),
        "non_us": total_rows - len(df_states),
        "pct_non_us": round((total_rows - len(df_states)) / total_rows * 100, 2) if total_rows > 0 else 0
    }

###################################################################################################
def filter_df(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    operator: str = "All",
    time_of_day: str = "All",
    aircraft: str = "All"
) -> pd.DataFrame:
    """Filter dataset by optional year range and categorical selections."""
    out = df.copy()

    if start_year is not None and end_year is not None:
        out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce")
        out = out[
            (out["YEAR"] >= start_year) &
            (out["YEAR"] <= end_year)
        ]

    if operator != "All":
        out = out[out["OPERATOR"] == operator]

    if time_of_day != "All":
        out = out[out["TIME_OF_DAY"] == time_of_day]

    if aircraft != "All":
        out = out[out["AIRCRAFT"] == aircraft]

    return out

###################################################################################################
def count_by(
    df: pd.DataFrame,
    col: str,
    label: str,
    n: Optional[int] = None,
    ascending: bool = False
) -> pd.DataFrame:
    """Group data by a column and return counts sorted by frequency."""
    out = (
        df.groupby(col, dropna=False)
        .size()
        .reset_index(name='Number of Strikes')
        .rename(columns={col: label})
        .sort_values('Number of Strikes', ascending=ascending)
    )

    out[label] = out[label].fillna('Unknown')

    if n is not None:
        out = out.head(n)
    
    return out

###################################################################################################
#Calculate count statistics
def get_top_counts(df_states: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    carriers_df = count_by(df_states, 'OPERATOR', 'Carrier', n=10)
    # Filter out Unknown before returning
    carriers_df = carriers_df[carriers_df["Carrier"] != "Unknown"]

    return {
        'states': count_by(df_states, 'REGION_CLEAN', 'State', n=10),
        'airports': count_by(df_states, 'AIRPORT', 'Airport', n=10),
        'carriers': carriers_df
    }

def get_time_counts(df_states: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    hour_df = count_by(df_states, "HOUR", "Hour")

    hour_df = hour_df[hour_df["Hour"] != "Unknown"].copy()
    hour_df["Hour"] = hour_df["Hour"].astype(int)
    hour_df = hour_df.sort_values("Hour")

    return {
        "time_of_day": count_by(df_states, "TIME_OF_DAY", "Time of Day"),
        "hour": hour_df,
        "phase": count_by(df_states, "PHASE_GROUP", "Phase of Flight"),
    }

def get_seasonality_counts(df_states: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Month
    month_df = count_by(df_states, "MONTH", "Month")
    month_df = month_df[month_df["Month"] != "Unknown"].copy()
    month_df["Month"] = month_df["Month"].astype(int)
    month_df = month_df.sort_values("Month")

    # Quarter
    quarter_df = count_by(df_states, "QUARTER", "Quarter")
    quarter_df = quarter_df[quarter_df["Quarter"] != "Unknown"].copy()
    quarter_df["Quarter"] = quarter_df["Quarter"].astype(int)
    quarter_df = quarter_df.sort_values("Quarter")

    return {
        "month": month_df,
        "quarter": quarter_df
    }

def get_aircraft_counts(df_states: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    num_engines_df = count_by(df_states, "NUM_ENGS", "Number of Engines")

    num_engines_df["Number of Engines"] = (
        num_engines_df["Number of Engines"]
        .replace({
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
        })
        .fillna("Unknown")
    )

    order = ["One", "Two", "Three", "Four", "Unknown"]
    num_engines_df["Number of Engines"] = pd.Categorical(
        num_engines_df["Number of Engines"],
        categories=order,
        ordered=True
    )

    num_engines_df = num_engines_df.sort_values("Number of Engines")

    return {
        "aircraft": count_by(df_states, "AIRCRAFT", "Aircraft", n=25),
        "engine_type": count_by(df_states, "ENGINE_TYPE", "Engine Type"),
        "num_engines": num_engines_df,
    }

def get_species_counts(df_states: pd.DataFrame) -> pd.DataFrame:
    return count_by(df_states, "SPECIES", "Species", n=25)


###################################################################################################
def get_yearly_counts(df_states: pd.DataFrame) -> pd.DataFrame:
    return (
        df_states.groupby("YEAR")
        .size()
        .reset_index(name="Number of Strikes")
        .rename(columns={"YEAR": "Year"})
        .sort_values("Year")
    )


###################################################################################################
# Heatmap
def get_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare latitude and longitude data for heatmap visualization."""
    out = df[["LATITUDE", "LONGITUDE"]].copy()
    out["LATITUDE"] = pd.to_numeric(out["LATITUDE"], errors="coerce")
    out["LONGITUDE"] = pd.to_numeric(out["LONGITUDE"], errors="coerce")
    return out.dropna(subset=["LATITUDE", "LONGITUDE"])


###################################################################################################
def get_scatter_data(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Prepare data for scatter plot with log transformations."""
    out = df[['TOTAL_COST', 'WARNED', x_col]].copy()

    # Convert to numeric
    out['TOTAL_COST'] = pd.to_numeric(out['TOTAL_COST'], errors='coerce')
    out[x_col] = pd.to_numeric(out[x_col], errors='coerce')

    out["WARNED"] = out["WARNED"].astype("category")

    # Log transform
    out['Log_Cost'] = np.log10(out['TOTAL_COST'] + 1)
    out[f'Log_{x_col}'] = np.log10(out[x_col] + 1)

    # Drop missing
    return out.dropna(subset = ['Log_Cost', f'Log_{x_col}'])


###################################################################################################
def compute_correlation(df: pd.DataFrame, x_col: str) -> float:
    if len(df) < 2:
        return np.nan
    
    if f"Log_{x_col}" not in df.columns:
        return np.nan
    
    r, _ = stats.pearsonr(df[f"Log_{x_col}"], df["Log_Cost"])
    return r