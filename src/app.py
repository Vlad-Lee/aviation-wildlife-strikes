import streamlit as st
from streamlit_option_menu import option_menu

from data import load_data

from app_pages.home import show_home
from app_pages.heatmap import show_heatmap
from app_pages.yearly_strikes import show_yearly_strikes
from app_pages.time_patterns import show_time
from app_pages.seasonality import show_seasonality
from app_pages.aircraft import show_aircraft
from app_pages.species import show_species
from app_pages.cost_relationships import show_scatter_plot


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Aviation Wildlife Strikes",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Load data
# -----------------------------
with st.spinner("Loading data..."):
    df = load_data().copy()

df_states = df[df["IS_US_STATE"] == 1].copy()

MIN_YEAR = int(df_states["YEAR"].min())

# Exclude incomplete current year from dashboard filters.
MAX_YEAR = 2025

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
            "Cost Relationships",
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
# About page
# -----------------------------
def show_about() -> None:
    st.title("About")

    st.markdown("""
        **Vlad Lee**  
        Data Scientist (MIDS, UC Berkeley)

        🔗 [LinkedIn](https://www.linkedin.com/in/vlad-lee)  
        💻 [GitHub Repository](https://github.com/Vlad-Lee/Aviation-Wildlife-Strikes)

        📧 vlad7984@gmail.com
    """)


# -----------------------------
# Page router
# -----------------------------
pages = {
    "Home": lambda: show_home(
        df,
        df_states,
        MIN_YEAR,
        MAX_YEAR,
    ),

    "Heatmap": lambda: show_heatmap(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "Yearly Strikes": lambda: show_yearly_strikes(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "Time": lambda: show_time(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "Seasonality": lambda: show_seasonality(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "Aircraft": lambda: show_aircraft(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "Species": lambda: show_species(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "Cost Relationships": lambda: show_scatter_plot(
        df_states,
        MIN_YEAR,
        MAX_YEAR,
        operator_list,
        time_of_day_list,
        aircraft_list,
    ),

    "About": show_about,
}


# -----------------------------
# Run selected page
# -----------------------------
pages[selected]()