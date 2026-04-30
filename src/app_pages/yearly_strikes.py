import matplotlib.pyplot as plt
import streamlit as st

from aggregations import get_yearly_counts
from filters import apply_filters, get_sidebar_filters
from plotting import show_plot, style_chart


def show_yearly_strikes(
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

    yearly_strikes_df = get_yearly_counts(filtered_df)

    if yearly_strikes_df.empty:
        st.warning("No wildlife strike records match the selected filters.")
        return

    years = yearly_strikes_df["Year"]

    colors = [
        "#f97316" if year == 2020 else "#60a5fa"
        for year in yearly_strikes_df["Year"]
    ]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        yearly_strikes_df["Year"],
        yearly_strikes_df["Number of Strikes"],
        color=colors,
    )

    covid_rows = yearly_strikes_df[yearly_strikes_df["Year"] == 2020]

    if not covid_rows.empty:
        covid_value = covid_rows["Number of Strikes"].iloc[0]

        ax.annotate(
            "COVID-19 drop",
            xy=(2020, covid_value),
            xytext=(2020, covid_value + 5000),
            ha="center",
            color="#f97316",
            fontsize=10,
            arrowprops={
                "arrowstyle": "->",
                "color": "#f97316",
                "lw": 1.5,
            },
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Wildlife Strikes")
    ax.set_xticks(years[::2])
    ax.set_xticklabels(years[::2])

    style_chart(ax)
    show_plot(fig)

    st.caption(
        "Wildlife strikes show a steady long-term increase, with a clear dip in 2020 due to reduced flight activity during COVID-19."
    )