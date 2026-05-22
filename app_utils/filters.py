from typing import Tuple, List
import pandas as pd
import streamlit as st

from app_utils.aggregations import filter_df

def get_sidebar_filters(
    min_year: int,
    max_year: int,
    operator_list: List[str],
    time_of_day_list: List[str],
    aircraft_list: List[str],
    include_operator: bool = True,
    include_time_of_day: bool = True,
    include_aircraft: bool = False,
    single_year: bool = False,
) -> Tuple[int, int, str, str, str]:
    """Create shared sidebar filters."""
    with st.sidebar:
        if single_year:
            selected_year = st.selectbox(
                "Select Year",
                options=list(range(min_year, max_year + 1)),
                index=max_year - min_year,
            )
            start_year, end_year = selected_year, selected_year
        else:
            year_range = st.slider(
                "Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1,
            )
            start_year, end_year = year_range

        selected_operator = "All"
        selected_time_of_day = "All"
        selected_aircraft = "All"

        if include_operator:
            selected_operator = st.selectbox("Select Operator", operator_list)

        if include_time_of_day:
            selected_time_of_day = st.selectbox("Select Time of Day", time_of_day_list)

        if include_aircraft:
            selected_aircraft = st.selectbox("Select Aircraft", aircraft_list)

    return start_year, end_year, selected_operator, selected_time_of_day, selected_aircraft


def apply_filters(
    df_states: pd.DataFrame,
    start_year: int,
    end_year: int,
    operator: str = "All",
    time_of_day: str = "All",
    aircraft: str = "All",
) -> pd.DataFrame:
    """Apply dashboard filters to the U.S. states dataframe."""
    return filter_df(
        df_states,
        start_year=start_year,
        end_year=end_year,
        operator=operator,
        time_of_day=time_of_day,
        aircraft=aircraft,
    )


def pct_value(df: pd.DataFrame, col: str, value: str = "Unknown") -> float:
    """Return percent of observations equal to a given value."""
    if col not in df.columns or df.empty:
        return 0.0

    return df[col].eq(value).mean() * 100