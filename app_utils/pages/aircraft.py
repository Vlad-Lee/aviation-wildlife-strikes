import matplotlib.pyplot as plt
import streamlit as st

from app_utils.aggregations import get_aircraft_counts
from app_utils.filters import apply_filters, get_sidebar_filters, pct_value
from app_utils.plotting import show_plot, style_chart


def show_aircraft(
    df_states,
    min_year: int,
    max_year: int,
    operator_list: list[str],
    time_of_day_list: list[str],
    aircraft_list: list[str],
) -> None:
    st.title("Wildlife Strikes by Aircraft Characteristics")

    start_year, end_year, selected_operator, selected_time_of_day, selected_aircraft = (
        get_sidebar_filters(
            min_year=min_year,
            max_year=max_year,
            operator_list=operator_list,
            time_of_day_list=time_of_day_list,
            aircraft_list=aircraft_list,
            include_operator=True,
            include_time_of_day=True,
            include_aircraft=True,
            single_year=True,
        )
    )

    filtered_df = apply_filters(
        df_states=df_states,
        start_year=start_year,
        end_year=end_year,
        operator=selected_operator,
        time_of_day=selected_time_of_day,
        aircraft=selected_aircraft,
    )

    if filtered_df.empty:
        st.warning("No wildlife strike records match the selected filters.")
        return
    
    st.caption(
        f"Wildlife strikes are concentrated among large commercial aircraft with turbofan engines. | {start_year} to {end_year}"
    )
    
    aircraft_counts = get_aircraft_counts(filtered_df)

    aircraft_df = aircraft_counts["aircraft"]
    engine_type_df = aircraft_counts["engine_type"]

    unknown_aircraft_pct = pct_value(filtered_df, "AIRCRAFT", "Unknown")
    unknown_engine_pct = pct_value(filtered_df, "ENGINE_TYPE", "Unknown")

    aircraft_df = aircraft_df[
        aircraft_df["Aircraft"] != "Unknown"
    ].copy()

    engine_type_df = engine_type_df[
        engine_type_df["Engine Type"] != "Unknown"
    ].copy()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Engine Type")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(engine_type_df["Engine Type"], engine_type_df["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Engine Type")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)

    with chart_col2:
        st.subheader("Aircraft Type")

        aircraft_plot_df = aircraft_df.sort_values("Number of Strikes", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(aircraft_plot_df["Aircraft"], aircraft_plot_df["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Aircraft")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)

    st.caption(
        f"Unknown values excluded. {unknown_aircraft_pct:.1f}% of records have unknown aircraft type and {unknown_engine_pct:.1f}% have unknown engine type."
    )