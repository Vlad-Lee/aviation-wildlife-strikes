import io
import zipfile
import requests

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
import streamlit as st

from aggregations import (
    get_summary_metrics,
    filter_df,
    get_top_counts,
    get_time_counts,
    get_seasonality_counts,
    get_aircraft_counts,
    get_species_counts,
    get_yearly_counts,
    get_heatmap_data,
    prepare_scatter_data,
    compute_correlation
)

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Aviation Wildlife Strikes",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data():
    url = "https://github.com/user-attachments/files/27153418/cleaned_wildlife_strikes.zip"
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    with z.open("cleaned_wildlife_strikes.parquet") as f:
        return pd.read_parquet(f)

def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# Load Data
# -----------------------------
df = load_data().copy()
df_states = df[df["IS_US_STATE"] == 1].copy()

MIN_YEAR = int(df_states["YEAR"].min())
MAX_YEAR = int(df_states["YEAR"].max())

operator_list = ["All"] + sorted(df_states["OPERATOR"].dropna().unique().tolist())
time_of_day_list = ["All"] + sorted(df_states["TIME_OF_DAY"].dropna().unique().tolist())
aircraft_list = ["All"] + sorted(df_states["AIRCRAFT"].dropna().unique().tolist())


# -----------------------------
# Sidebar menu
# -----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=[
            "Home",
            "Heatmap",
            "Yearly Strikes",
            "Time",
            "Seasonality",
            "Aircraft",
            "Species",
            "Scatter Plot",
            "About",
        ],
        icons=[
            "house",
            "map",
            "calendar",
            "clock",
            "tropical-storm",
            "airplane",
            "feather",
            "graph-up",
            "question",
        ],
        menu_icon="cast",
        default_index=0,
    )


# -----------------------------
# Home
# -----------------------------
if selected == "Home":
    st.title("Wildlife Strikes in Aviation Dashboard")
    st.markdown(
        """
        Welcome to the **Wildlife Strikes in Aviation Dashboard**! This app allows users to analyze and visualize aviation wildlife strike data from **1990 to present**.

        👉 **Get started** by selecting an option from the sidebar.

        Data obtained from the official 
        [**FAA Wildlife Strike Database**](https://wildlife.faa.gov/).
        """,
        unsafe_allow_html=True,
    )

    metrics = get_summary_metrics(df, df_states)

    st.markdown(f"## Key Metrics ({MIN_YEAR} - {MAX_YEAR})")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Wildlife Strikes", f"{metrics['total_bird_strikes']:,}")

    with col2:
        st.metric("Total Cost", f"${metrics['total_cost']:,.0f}")

    with col3:
        st.metric(
            "Total Cost (Inflation Adjusted)",
            f"${metrics['total_cost_infl_adj']:,.0f}",
        )

    with col4:
        st.metric("Total Fatalities", f"{metrics['total_fatalities']:,.0f}")

    with col5:
        st.metric("Total Injuries", f"{metrics['total_injuries']:,.0f}")

    with col6:
        st.metric("Non-US (%)", f"{metrics['pct_non_us']:.2f}%")

    counts = get_top_counts(df_states)

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("Top 10 States with Wildlife Strikes")
        state_count = counts["states"]

        fig, ax = plt.subplots()
        ax.barh(state_count["State"], state_count["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("State")
        ax.invert_yaxis()
        show_plot(fig)

    with chart_col2:
        st.subheader("Top 10 Airports with Wildlife Strikes")
        airport_count = counts["airports"]

        fig, ax = plt.subplots()
        ax.barh(airport_count["Airport"], airport_count["Number of Strikes"], color='green')
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Airport")
        ax.invert_yaxis()
        show_plot(fig)

    with chart_col3:
        st.subheader("Top 10 Carriers with Wildlife Strikes")
        carrier_count = counts["carriers"]

        fig, ax = plt.subplots()
        ax.barh(carrier_count["Carrier"], carrier_count["Number of Strikes"], color='orange')
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Carrier")
        ax.invert_yaxis()
        show_plot(fig)


# -----------------------------
# Heatmap
# -----------------------------
elif selected == "Heatmap":
    st.title("Heatmap of U.S. Wildlife Strikes")

    with st.sidebar:
        year_range = st.slider(
            "Select Year Range",
            min_value=MIN_YEAR,
            max_value=MAX_YEAR,
            value=(MIN_YEAR, MAX_YEAR),
            step=1,
        )

        selected_operator = st.selectbox("Select Operator", operator_list)
        selected_time_of_day = st.selectbox("Select Time of Day", time_of_day_list)

    start_year, end_year = year_range

    filtered_df = filter_df(
        df_states,
        start_year=start_year,
        end_year=end_year,
        operator=selected_operator,
        time_of_day=selected_time_of_day,
    )

    heatmap_df = get_heatmap_data(filtered_df)

    st.markdown(f"### U.S. Wildlife Strikes ({start_year} - {end_year})")

    if heatmap_df.empty:
        st.warning("No wildlife strike records match the selected filters.")
    else:
        us_map = folium.Map(location=[39.5, -98.35], zoom_start=4)
        heat_data = heatmap_df[["LATITUDE", "LONGITUDE"]].values.tolist()
        HeatMap(heat_data, radius=8).add_to(us_map)
        folium_static(us_map)


# -----------------------------
# Yearly Strikes
# -----------------------------
elif selected == "Yearly Strikes":
    st.title("Yearly Number of Wildlife Strikes")

    with st.sidebar:
        year_range = st.slider(
            "Select Year Range",
            min_value=MIN_YEAR,
            max_value=MAX_YEAR,
            value=(MIN_YEAR, MAX_YEAR),
            step=1,
        )

        selected_operator = st.selectbox("Select Operator", operator_list)
        selected_time_of_day = st.selectbox("Select Time of Day", time_of_day_list)
        selected_aircraft = st.selectbox("Select Aircraft", aircraft_list)

    start_year, end_year = year_range

    filtered_df = filter_df(
        df_states,
        start_year=start_year,
        end_year=end_year,
        operator=selected_operator,
        time_of_day=selected_time_of_day,
        aircraft=selected_aircraft,
    )

    yearly_strikes_df = get_yearly_counts(filtered_df)

    if yearly_strikes_df.empty:
        st.warning("No wildlife strike records match the selected filters.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(yearly_strikes_df["Year"], yearly_strikes_df["Number of Strikes"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.tick_params(axis="x", rotation=45)
        show_plot(fig)

# -----------------------------
# Time
# -----------------------------
elif selected == "Time":
    st.title("Wildlife Strikes by Time and Flight Phase")

    time_counts = get_time_counts(df_states)

    time_of_day_df = time_counts["time_of_day"]
    hourly_df = time_counts["hour"]
    phase_df = time_counts["phase"]

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("By Time of Day")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            time_of_day_df["Number of Strikes"],
            labels=time_of_day_df["Time of Day"],
            autopct="%1.1f%%",
            startangle=90,
        )
        show_plot(fig)

    with chart_col2:
        st.subheader("By Hour")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.bar(hourly_df["Hour"], hourly_df["Number of Strikes"])
        ax.set_xlabel("Hour")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xticks(hourly_df["Hour"])
        ax.tick_params(axis="x", rotation=45)
        show_plot(fig)

    with chart_col3:
        st.subheader("By Phase of Flight")

        fig, ax = plt.subplots()
        ax.barh(phase_df["Phase of Flight"], phase_df["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Phase of Flight")
        ax.invert_yaxis()
        show_plot(fig)
        
# -----------------------------
# Seasonality
# -----------------------------
elif selected == "Seasonality":
    st.title("Seasonality Analysis")

    seasonality_counts = get_seasonality_counts(df_states)

    monthly_strikes_df = seasonality_counts["month"]
    quarterly_strikes_df = seasonality_counts["quarter"]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Monthly Wildlife Strikes")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            monthly_strikes_df["Month"],
            monthly_strikes_df["Number of Strikes"],
            width=0.6,
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        show_plot(fig)

    with chart_col2:
        st.subheader("Quarterly Wildlife Strikes")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            quarterly_strikes_df["Quarter"],
            quarterly_strikes_df["Number of Strikes"],
            width=0.6,
        )
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        show_plot(fig)
       
# -----------------------------
# Aircraft
# -----------------------------
elif selected == "Aircraft":
    st.title("Wildlife Strikes by Aircraft Characteristics")

    aircraft_counts = get_aircraft_counts(df_states)

    aircraft_df = aircraft_counts["aircraft"]
    engine_type_df = aircraft_counts["engine_type"]
    num_engines_df = aircraft_counts["num_engines"]

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("By Aircraft Type")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(aircraft_df["Aircraft"], aircraft_df["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Aircraft")
        ax.invert_yaxis()
        show_plot(fig)

    with chart_col2:
        st.subheader("By Engine Type")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(engine_type_df["Engine Type"], engine_type_df["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Engine Type")
        ax.invert_yaxis()
        show_plot(fig)

    with chart_col3:
        st.subheader("By Number of Engines")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(num_engines_df["Number of Engines"], num_engines_df["Number of Strikes"])
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Number of Engines")
        ax.invert_yaxis()
        show_plot(fig)

# -----------------------------
# Species
# -----------------------------
elif selected == "Species":
    st.title("Number of Wildlife Strikes by Species")

    species_df = get_species_counts(df_states)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(species_df["Species"], species_df["Number of Strikes"])
    ax.set_xlabel("Number of Wildlife Strikes")
    ax.set_ylabel("Species")
    ax.invert_yaxis()
    show_plot(fig)


# -----------------------------
# Scatter Plot
# -----------------------------
elif selected == "Scatter Plot":
    st.title("Scatter Plot: Cost vs Flight Variables")

    with st.sidebar:
        year_range = st.slider(
            "Select Year Range",
            min_value=MIN_YEAR,
            max_value=MAX_YEAR,
            value=(MIN_YEAR, MAX_YEAR),
            step=1,
        )

        x_variable = st.selectbox(
            "Select X Variable",
            options=["Height", "Speed", "Distance"],
            index=0,
        )

    start_year, end_year = year_range

    x_column_map = {
        "Height": "HEIGHT",
        "Speed": "SPEED",
        "Distance": "DISTANCE",
    }
    x_col = x_column_map[x_variable]

    filtered_df = filter_df(
        df_states,
        start_year=start_year,
        end_year=end_year,
    )

    scatter_df = prepare_scatter_data(filtered_df, x_col)

    if scatter_df.empty:
        st.warning("No records match the selected filters.")
    else:
        pearson_r = compute_correlation(scatter_df, x_col)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=scatter_df,
            x=f"Log_{x_col}",
            y="Log_Cost",
            hue="WARNED",
            ax=ax,
        )

        ax.set_xlabel(f"Log {x_variable}")
        ax.set_ylabel("Log Cost")
        ax.set_title(f"Log Cost vs Log {x_variable} | Pearson's r = {pearson_r:.2f}")
        show_plot(fig)
       
# -----------------------------
# About
# -----------------------------
elif selected == "About":
    st.title("About")

    st.markdown("""
        **Vlad Lee**<br>
        Data Scientist (MIDS, UC Berkeley)<br><br>

        🔗 <a href="https://www.linkedin.com/in/vlad-lee" target="_blank">LinkedIn</a><br>
        💻 <a href="https://github.com/Vlad-Lee/Aviation-Wildlife-Strikes" target="_blank">GitHub Repository</a><br><br>

        For questions or feedback:<br>
        📧 <a href="mailto:Vlad7984@gmail.com">vlad7984@gmail.com</a>
    """, unsafe_allow_html=True)