import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# Import styling utilities
from app_utils.plotting import style_chart

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODELS_DIR / "stage1_classifier.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_height_bin(height):
    if height == 0:      return "GROUND"
    if height < 500:     return "LOW"
    if height < 3000:    return "APPROACH_CLIMB"
    return "CRUISE"


def get_risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.30:  return "🟢 Low Risk",      "#22c55e"
    if prob < 0.60:  return "🟡 Moderate Risk",  "#eab308"
    return               "🔴 High Risk",          "#ef4444"


def plot_gauge(prob: float) -> plt.Figure:
    """
    Draw a semicircular gauge from 0% to 100%.
    The needle points to the current probability.
    Zones: green 0-30%, yellow 30-60%, red 60-100%.
    """
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    def prob_to_theta(p):
        return np.pi * (1 - p)

    zones = [
        (0.00, 0.30, "#22c55e"),
        (0.30, 0.60, "#eab308"),
        (0.60, 1.00, "#ef4444"),
    ]

    bar_width = 0.35
    for p_start, p_end, color in zones:
        t_start = prob_to_theta(p_end)
        t_end   = prob_to_theta(p_start)
        theta = np.linspace(t_start, t_end, 100)
        ax.fill_between(theta, 0.65, 0.65 + bar_width, color=color, alpha=0.85)

    needle_theta = prob_to_theta(prob)
    ax.annotate(
        "",
        xy=(needle_theta, 0.62),
        xytext=(needle_theta, 0.05),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
    )

    ax.plot(needle_theta, 0.05, "o", color="white", markersize=6, zorder=5)

    ax.text(
        np.pi / 2, 0.28,
        f"{prob * 100:.1f}%",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="white",
        transform=ax.transData,
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.axis("off")

    return fig


# ---------------------------------------------------------
# Main Render
# ---------------------------------------------------------
def show_prediction(df_states, aircraft_list):
    st.title("Strike Risk Predictor")
    st.markdown("Adjust the parameters below to estimate the probability of financial damage from a wildlife strike.")

    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
            This predictor uses a **Random Forest classifier** trained on FAA Wildlife Strike Database
            records from 1990–2025. It predicts the probability that a given strike results in
            reportable financial damage.

            The model uses pre-event flight and wildlife parameters as features: aircraft speed,
            altitude, wildlife size (mapped to a kinetic energy proxy), aircraft mass class, engine
            type, operator category, and FAA region. These are all observable before or at the moment
            of a strike, making this a genuine pre-event risk assessment tool.

            Training used stratified cross-validation with Optuna hyperparameter tuning, and the
            classification threshold was calibrated on the held-out test set to maximize F1 score
            rather than defaulting to 0.50.
        """)

    clf = load_model()
    if not clf:
        return

    st.write("---")

    # ==========================================
    # UI Section 1: Inputs
    # ==========================================
    input_col1, input_col2 = st.columns(2, gap="large")

    with input_col1:
        st.subheader("Flight & Wildlife Parameters")
        speed     = st.slider("Aircraft Speed (knots)", 0, 400, 150)
        height    = st.slider("Altitude (ft AGL)", 0, 35000, 1000)
        size_full = st.selectbox("Wildlife Size", [
            "Small (e.g. sparrow, pigeon)",
            "Medium (e.g. gull, hawk)",
            "Large (e.g. goose, eagle)",
        ])
        size_code = size_full.split(" ")[0].upper()

    with input_col2:
        st.subheader("Aircraft Profile")

        operator_cat_full = st.selectbox("Operator Category", [
            "BUS (Business/Commercial - Airlines, cargo, air taxis)",
            "GOV (Government - Non-military agencies like Coast Guard)",
            "MIL (Military - Armed forces aircraft)",
            "PVT (Private - General aviation, corporate jets)",
        ])

        ac_mass_full = st.selectbox("Aircraft Mass Class", [
            "2.0 (Light: 2,251 - 12,500 lbs)",
            "3.0 (Medium: 12,501 - 119,000 lbs)",
            "4.0 (Heavy: 119,001 - 850,000 lbs)",
            "5.0 (Jumbo: Over 850,000 lbs)",
        ])

        engine_type_full = st.selectbox("Engine Type", [
            "B (Turbojet - Older jet engines)",
            "C (Turboprop - Propeller driven by turbine)",
            "D (Turbofan - Modern commercial airliners)",
            "E (None - Gliders)",
            "F (Turboshaft - Helicopters)",
            "Y (Other / Unknown)",
        ])

    # ==========================================
    # UI Section 2: Data Processing & Logic
    # ==========================================
    operator_code    = operator_cat_full.split(" ")[0]
    ac_mass_code     = float(ac_mass_full.split(" ")[0])
    engine_type_code = engine_type_full.split(" ")[0]

    size_mass_map = {"SMALL": 1, "MEDIUM": 5, "LARGE": 15}
    ke_joules = size_mass_map[size_code] * (speed ** 2)

    input_dict = {
        "SPEED":              float(speed),
        "LATITUDE":           39.0,
        "LONGITUDE":          -95.0,
        "NUM_ENGS":           2.0,
        "MONTH":              6,
        "HAS_PRECIP":         0,
        "IS_MULTI_STRIKE":    0,
        "LOG_KINETIC_ENERGY": float(np.log1p(ke_joules)),
        "WARNED_FLAG":        0,
        "AC_MASS":            ac_mass_code,
        "HEIGHT_BIN":         get_height_bin(height),
        "OPERATOR_CAT":       operator_code,
        "ENGINE_TYPE":        engine_type_code,
        "FAAREGION":          "EA",
    }

    input_df = pd.DataFrame([input_dict])
    cat_cols = ["AC_MASS", "HEIGHT_BIN", "OPERATOR_CAT", "ENGINE_TYPE", "FAAREGION"]
    input_df = pd.get_dummies(input_df, columns=cat_cols)

    try:
        input_df = input_df.reindex(columns=clf.feature_names_in_, fill_value=0)
    except AttributeError:
        st.error("Model missing feature_names_in_")
        return

    prob_damage = 0.0
    try:
        damage_probs = clf.predict_proba(input_df)
        classes = list(clf.classes_)
        if 1 in classes:
            prob_damage = float(damage_probs.flatten()[classes.index(1)])
        elif "1" in classes:
            prob_damage = float(damage_probs.flatten()[classes.index("1")])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # ==========================================
    # UI Section 3: Results
    # ==========================================
    st.write("---")
    st.subheader("Prediction Results")

    risk_label, risk_color = get_risk_label(prob_damage)

    gauge_col, label_col = st.columns([1, 1], gap="large")

    with gauge_col:
        st.markdown("**Damage Probability**")
        gauge_fig = plot_gauge(prob_damage)
        st.pyplot(gauge_fig, use_container_width=True)

    with label_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='color:{risk_color};'>{risk_label}</h2>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Probability that this strike results in reportable financial damage, "
            "based on pre-event flight and wildlife parameters."
        )

    # ==========================================
    # UI Section 4: Physics & Feature Importance
    # ==========================================
    with st.expander("🔍 View Physics & Model Analysis", expanded=True):

        phys_col, fi_col = st.columns(2, gap="large")

        with phys_col:
            st.markdown("**Kinetic Impact Energy**")
            st.markdown(
                f"This scenario generated a kinetic energy proxy of **{ke_joules:,.0f}** for this strike.  \n"
                f"*KE Proxy = size mass × speed²*  \n"
                f"Kinetic energy is the strongest single predictor of damage probability "
                f"(importance score ~0.24)."
            )

            max_display = max(ke_joules * 2, 10_000)
            pct = min(ke_joules / max_display, 1.0)
            bar_color = (
                "#22c55e" if pct < 0.33
                else "#eab308" if pct < 0.66
                else "#ef4444"
            )

            fig_e, ax_e = plt.subplots(figsize=(6, 1.2))
            fig_e.patch.set_alpha(0)
            ax_e.set_facecolor("none")
            ax_e.barh([""], [ke_joules], color=bar_color)
            ax_e.set_xlim(0, max_display)
            ax_e.set_xlabel("Joules")
            style_chart(ax_e)
            st.pyplot(fig_e, use_container_width=True)

        with fi_col:
            st.markdown("**Top Model Features**")
            clf_img = FIGURES_DIR / "stage_1_classifier_importance.png"
            if clf_img.exists():
                st.image(str(clf_img), use_container_width=True)
            else:
                st.warning(f"Image not found: {clf_img}")