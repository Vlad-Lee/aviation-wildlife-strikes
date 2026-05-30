import streamlit as st
from streamlit_option_menu import option_menu

from app_utils.data import load_data
from app_utils.pages.home import show_home
from app_utils.pages.heatmap import show_heatmap
from app_utils.pages.yearly_strikes import show_yearly_strikes
from app_utils.pages.time_patterns import show_time
from app_utils.pages.seasonality import show_seasonality
from app_utils.pages.aircraft import show_aircraft
from app_utils.pages.species import show_species
from app_utils.pages.prediction import show_prediction

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
MAX_YEAR = 2025

operator_list = ["All"] + sorted(df_states["OPERATOR"].dropna().unique().tolist())
time_of_day_list = ["All"] + sorted(df_states["TIME_OF_DAY"].dropna().unique().tolist())
aircraft_list = ["All"] + sorted(df_states["AIRCRAFT"].dropna().unique().tolist())

# -----------------------------
# About page
# -----------------------------
def show_about() -> None:
    st.title("About")
    st.markdown("""
        **Vlad Lee** Data Scientist (MIDS, UC Berkeley)

        🔗 [LinkedIn](https://www.linkedin.com/in/vlad-lee)  
        
        💻 [GitHub Repository](https://github.com/Vlad-Lee/Aviation-Wildlife-Strikes)

        📧 vlad7984@gmail.com
    """)

# -----------------------------
# Sidebar menu
# -----------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=[
            "Home",
            "Damage Predictor",
            "Heatmap",
            "Yearly Strikes",
            "Time",
            "Seasonality",
            "Aircraft",
            "Species",
            "About",
        ],
        icons=[
            "house",
            "calculator",
            "map",
            "calendar",
            "clock",
            "tropical-storm",
            "airplane",
            "feather",
            "question",
        ],
        menu_icon="cast",
        default_index=0,
    )

# -----------------------------
# Page router
# -----------------------------
pages = {
    "Home": lambda: show_home(df, df_states, MIN_YEAR, MAX_YEAR),
    
    "Damage Predictor": lambda: show_prediction(df_states, aircraft_list),

    "Heatmap": lambda: show_heatmap(df_states, MIN_YEAR, MAX_YEAR, operator_list, time_of_day_list, aircraft_list),

    "Yearly Strikes": lambda: show_yearly_strikes(df_states, MIN_YEAR, MAX_YEAR, operator_list, time_of_day_list, aircraft_list),

    "Time": lambda: show_time(df_states, MIN_YEAR, MAX_YEAR, operator_list, time_of_day_list, aircraft_list),

    "Seasonality": lambda: show_seasonality(df_states, MIN_YEAR, MAX_YEAR, operator_list, time_of_day_list, aircraft_list),

    "Aircraft": lambda: show_aircraft(df_states, MIN_YEAR, MAX_YEAR, operator_list, time_of_day_list, aircraft_list),

    "Species": lambda: show_species(df_states, MIN_YEAR, MAX_YEAR, operator_list, time_of_day_list, aircraft_list),

    "About": show_about,
}

# -----------------------------
# Run selected page
# -----------------------------
pages[selected]()
