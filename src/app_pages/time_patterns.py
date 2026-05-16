import matplotlib.pyplot as plt
import streamlit as st

from aggregations import get_time_counts
from filters import apply_filters, get_sidebar_filters, pct_value
from plotting import show_plot, style_chart


def show_time(
    df_states,
    min_year: int,
    max_year: int,
    operator_list: list[str],
    time_of_day_list: list[str],
    aircraft_list: list[str],
) -> None:
    st.title("When Wildlife Strikes Occur")

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
            single_year=False,
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
        f"Wildlife strikes are most common during daytime hours and during approach, takeoff, and landing. | {start_year} to {end_year}"
    )

    time_counts = get_time_counts(filtered_df)

    time_of_day_df = time_counts["time_of_day"]
    hourly_df = time_counts["hour"]
    phase_df = time_counts["phase"]

    unknown_time_pct = pct_value(filtered_df, "TIME_OF_DAY")
    unknown_phase_pct = pct_value(filtered_df, "PHASE_GROUP")

    time_of_day_df = time_of_day_df[
        time_of_day_df["Time of Day"] != "Unknown"
    ].copy()

    phase_df = phase_df[
        phase_df["Phase of Flight"] != "Unknown"
    ].copy()

    time_of_day_df = time_of_day_df.sort_values("Time of Day")

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("Time of Day")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(
            time_of_day_df["Time of Day"],
            time_of_day_df["Number of Strikes"],
            color="#60a5fa",
        )
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Time of Day")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)

    with chart_col2:
        st.subheader("Hour of Day")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.bar(
            hourly_df["Hour"],
            hourly_df["Number of Strikes"],
            color="#34d399",
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xticks(hourly_df["Hour"])
        ax.tick_params(axis="x", rotation=45)
        style_chart(ax)
        show_plot(fig)

    with chart_col3:
        st.subheader("Phase of Flight")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(
            phase_df["Phase of Flight"],
            phase_df["Number of Strikes"],
            color="#f97316",
        )
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Phase of Flight")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)
    