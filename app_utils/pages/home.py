import matplotlib.pyplot as plt
import streamlit as st

from app_utils.aggregations import get_summary_metrics, get_top_counts
from app_utils.plotting import show_plot, style_chart


def show_home(df, df_states, min_year: int, max_year: int) -> None:
    st.title("Wildlife Strikes in Aviation Dashboard")

    st.markdown(
        f"""
        Welcome to the **Wildlife Strikes in Aviation Dashboard**! This app allows users
        to analyze and visualize aviation wildlife strike data from **{min_year} to {max_year}**.

        👉 **Get started** by selecting an option from the sidebar.

        Data obtained from the official
        [**FAA Wildlife Strike Database**](https://wildlife.faa.gov/).
        """,
        unsafe_allow_html=True,
    )

    st.caption("This dashboard focuses on wildlife strikes occurring within U.S. states.")

    # -----------------------------
    # Summary metrics
    # -----------------------------
    metrics = get_summary_metrics(df, df_states)

    st.markdown(f"## Key Metrics ({min_year} - {max_year})")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Wildlife Strikes", f"{metrics['total_bird_strikes']:,}")
        st.metric("Total Fatalities", f"{metrics['total_fatalities']:,.0f}")

    with col2:
        st.metric("Damaged Aircraft", f"{metrics['total_damaged_aircraft']:,}")
        st.metric("Total Injuries", f"{metrics['total_injuries']:,.0f}")

    with col3:
        st.metric("Structural Damage Rate", f"{metrics['damage_rate_pct']:.1f}%")
        st.metric("Non-US Incidents", f"{metrics['pct_non_us']:.1f}%")
        
    with col4:
        st.metric("Total Cost (Inflation Adj)", f"${metrics['total_cost_infl_adj']:,.0f}")
        st.metric("Median Strike Cost (Inflation Adj)", f"${metrics['median_cost_infl_adj']:,.0f}")

    # -----------------------------
    # Top counts
    # -----------------------------
    counts = get_top_counts(df_states)

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # States
    with chart_col1:
        st.subheader("Top 10 States with Wildlife Strikes")
        state_count = counts["states"]

        fig, ax = plt.subplots()
        ax.barh(
            state_count["State"],
            state_count["Number of Strikes"],
            color="#60a5fa",
        )
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("State")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)

    # Airports
    with chart_col2:
        st.subheader("Top 10 Airports with Wildlife Strikes")
        airport_count = counts["airports"]

        fig, ax = plt.subplots()
        ax.barh(
            airport_count["Airport"],
            airport_count["Number of Strikes"],
            color="#34d399",
        )
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Airport")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)

    # Carriers
    with chart_col3:
        st.subheader("Top 10 Carriers with Wildlife Strikes")
        carrier_count = counts["carriers"]

        fig, ax = plt.subplots()
        ax.barh(
            carrier_count["Carrier"],
            carrier_count["Number of Strikes"],
            color="#f97316",
        )
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Carrier")
        ax.invert_yaxis()
        style_chart(ax)
        show_plot(fig)