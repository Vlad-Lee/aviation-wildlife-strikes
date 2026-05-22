import folium
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import folium_static

from app_utils.aggregations import get_heatmap_data
from app_utils.filters import apply_filters, get_sidebar_filters


def show_heatmap(
    df_states,
    min_year: int,
    max_year: int,
    operator_list: list[str],
    time_of_day_list: list[str],
    aircraft_list: list[str],
) -> None:
    st.title("Heatmap of U.S. Wildlife Strikes")

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

    heatmap_df = get_heatmap_data(filtered_df)

    heatmap_df = heatmap_df[
        heatmap_df["LATITUDE"].between(24, 50)
        & heatmap_df["LONGITUDE"].between(-125, -66)
    ]

    st.caption(f"Showing results for {start_year}.")

    if heatmap_df.empty:
        st.warning("No wildlife strike records match the selected filters.")
        return

    us_map = folium.Map(location=[39.5, -98.35], zoom_start=4)
    heat_data = heatmap_df[["LATITUDE", "LONGITUDE"]].values.tolist()

    HeatMap(heat_data, radius=8).add_to(us_map)
    folium_static(us_map)

    st.caption(
        "Heatmap shows wildlife strikes in the contiguous United States. Alaska and Hawaii are excluded."
    )