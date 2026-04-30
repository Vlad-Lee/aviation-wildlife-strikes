import matplotlib.pyplot as plt
import streamlit as st

from aggregations import get_species_counts
from filters import apply_filters, get_sidebar_filters
from plotting import show_plot, style_chart


def show_species(
    df_states,
    min_year: int,
    max_year: int,
    operator_list: list[str],
    time_of_day_list: list[str],
    aircraft_list: list[str],
) -> None:
    st.title("Number of Wildlife Strikes by Species")

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
        f"A small number of species account for a large share of reported wildlife strikes. | {start_year} to {end_year}"
    )

    species_df = get_species_counts(filtered_df)
    species_df = species_df.head(15)

    species_df = get_species_counts(filtered_df)

    if species_df.empty:
        st.warning("No wildlife strike records match the selected filters.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(species_df["Species"], species_df["Number of Strikes"])
    ax.set_xlabel("Number of Wildlife Strikes")
    ax.set_ylabel("Species")
    ax.invert_yaxis()
    style_chart(ax)
    show_plot(fig)