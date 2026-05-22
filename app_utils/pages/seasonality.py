import matplotlib.pyplot as plt
import streamlit as st

from app_utils.aggregations import get_seasonality_counts
from app_utils.filters import apply_filters, get_sidebar_filters
from app_utils.plotting import show_plot, style_chart


def show_seasonality(
    df_states,
    min_year: int,
    max_year: int,
    operator_list: list[str],
    time_of_day_list: list[str],
    aircraft_list: list[str],
) -> None:
    st.title("Seasonality Analysis")

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

    seasonality_counts = get_seasonality_counts(filtered_df)

    monthly_strikes_df = seasonality_counts["month"]
    quarterly_strikes_df = seasonality_counts["quarter"]

    month_colors = [
        "#f97316" if month == 8 else "#60a5fa"
        for month in monthly_strikes_df["Month"]
    ]

    quarter_colors = [
        "#f97316" if quarter == 3 else "#60a5fa"
        for quarter in quarterly_strikes_df["Quarter"]
    ]

    st.caption(
        f"Wildlife strikes peak in late summer, especially in August and Q3. "
        f"Year range: {start_year} to {end_year}"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Monthly Wildlife Strikes")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            monthly_strikes_df["Month"],
            monthly_strikes_df["Number of Strikes"],
            width=0.6,
            color=month_colors,
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        style_chart(ax)
        show_plot(fig)

    with chart_col2:
        st.subheader("Quarterly Wildlife Strikes")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            quarterly_strikes_df["Quarter"],
            quarterly_strikes_df["Number of Strikes"],
            width=0.6,
            color=quarter_colors,
        )
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        style_chart(ax)
        show_plot(fig)

    st.caption(
        "Strike counts are highest in late summer and early fall, consistent with seasonal wildlife movement and higher bird activity."
    )