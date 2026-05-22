import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "output" / "figures"


@st.cache_resource
def load_ml_models():
    """Load models safely into memory."""
    try:
        stage1 = joblib.load(MODELS_DIR / "stage1_classifier.joblib")
        stage2 = joblib.load(MODELS_DIR / "stage2_regressor.joblib")
        return stage1, stage2
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def get_height_bin(height):
    """Categorize altitude into flight phases."""
    if height == 0:
        return "GROUND"
    if height < 500:
        return "LOW"
    if height < 3000:
        return "APPROACH_CLIMB"
    return "CRUISE"


def get_risk_label(prob: float) -> tuple[str, str]:
    """Return (label, color) based on damage probability."""
    if prob < 0.30:
        return "🟢 Low Risk", "#22c55e"
    if prob < 0.60:
        return "🟡 Moderate Risk", "#eab308"
    return "🔴 High Risk", "#ef4444"


def plot_gauge(prob: float) -> plt.Figure:
    """
    Draw a semicircular gauge from 0% to 100%.
    The needle points to the current probability.
    Zones: green 0-30%, yellow 30-60%, red 60-100%.
    """
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Gauge spans π to 0 (left to right, top semicircle)
    theta_min, theta_max = np.pi, 0.0

    def prob_to_theta(p):
        return np.pi * (1 - p)

    # Zone boundaries in probability space
    zones = [
        (0.00, 0.30, "#22c55e"),
        (0.30, 0.60, "#eab308"),
        (0.60, 1.00, "#ef4444"),
    ]

    bar_width = 0.35
    for p_start, p_end, color in zones:
        t_start = prob_to_theta(p_end)   # theta decreases as prob increases
        t_end   = prob_to_theta(p_start)
        theta = np.linspace(t_start, t_end, 100)
        ax.fill_between(theta, 0.65, 0.65 + bar_width, color=color, alpha=0.85)

    # Needle
    needle_theta = prob_to_theta(prob)
    ax.annotate(
        "",
        xy=(needle_theta, 0.62),
        xytext=(needle_theta, 0.05),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
    )

    # Centre dot
    ax.plot(needle_theta, 0.05, "o", color="white", markersize=6, zorder=5)

    # Probability label in centre
    ax.text(
        np.pi / 2, 0.28,
        f"{prob * 100:.1f}%",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="white",
        transform=ax.transData,
    )

    # Axis cosmetics
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
    median_cost_infl_adj = float(
        df_states[df_states["TOTAL_COST_INFL_ADJ"] > 0]["TOTAL_COST_INFL_ADJ"].median()
    )
    
    st.title("Strike Cost Predictor")
    st.markdown("Adjust the parameters below to simulate a wildlife strike and estimate the financial impact.")

    clf, reg = load_ml_models()
    if not clf or not reg:
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
        size_code = size_full.split(" ")[0].upper()  # "SMALL", "MEDIUM", "LARGE"

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
    operator_code    = operator_cat_full.split(" ")[0]        # e.g. "BUS"
    ac_mass_code     = float(ac_mass_full.split(" ")[0])      # e.g. 2.0
    engine_type_code = engine_type_full.split(" ")[0]         # e.g. "D"

    # Kinetic energy
    size_mass_map = {"SMALL": 1, "MEDIUM": 5, "LARGE": 15}
    ke_proxy = size_mass_map[size_code] * (speed ** 2)
    ke_joules = ke_proxy

    input_dict = {
        "SPEED":              float(speed),
        "LATITUDE":           39.0,
        "LONGITUDE":          -95.0,
        "NUM_ENGS":           2.0,
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
        model_cols = clf.feature_names_in_
        input_df = input_df.reindex(columns=model_cols, fill_value=0)
    except AttributeError:
        st.error("Model missing feature_names_in_")
        return

    # Stage 1 — damage probability
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

    # Stage 2 — estimated cost (only when damage likely)
    cost = None
    if prob_damage >= 0.50:
        try:
            log_cost = reg.predict(input_df)
            cost = float(np.expm1(log_cost.item()))
        except Exception as e:
            st.error(f"Cost prediction failed: {e}")

    # ==========================================
    # UI Section 3: Results
    # ==========================================
    st.write("---")
    st.subheader("Prediction Results")

    risk_label, risk_color = get_risk_label(prob_damage)

    gauge_col, results_col = st.columns([1, 1], gap="large")

    with gauge_col:
        st.markdown("**Damage Probability**")
        gauge_fig = plot_gauge(prob_damage)
        st.pyplot(gauge_fig, use_container_width=True)
        st.markdown(
            f"<h3 style='text-align:center; color:{risk_color};'>{risk_label}</h3>",
            unsafe_allow_html=True,
        )

    with results_col:
        st.markdown("**Estimated Cost**")
        if cost is not None:
            delta_vs_median = cost - median_cost_infl_adj
            delta_str = (
                f"+${delta_vs_median:,.0f} vs. median"
                if delta_vs_median >= 0
                else f"-${abs(delta_vs_median):,.0f} vs. median"
            )
            st.metric(
                label="Predicted Cost (Inflation Adj)",
                value=f"${cost:,.0f}",
                delta=delta_str,
                delta_color="inverse",
            )
            st.caption(
                f"Median damaging-strike cost (inflation adj): **${median_cost_infl_adj:,.0f}**  \n"
                "Based on FAA Wildlife Strike Database records where damage was reported."
            )
        else:
            st.metric("Predicted Cost", "$0")
        
        st.caption("Cost is only estimated when damage probability exceeds 50%.")  # ← always shown

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
                f"Kinetic energy is the single strongest predictor in both model stages "
                f"(importance score ~0.24 for damage classification, ~0.33 for cost estimation)."
            )

            # Gauge-style energy bar — colour scales with magnitude, no hardcoded threshold
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
            clf_img  = FIGURES_DIR / "stage_1_classifier_importance.png"
            reg_img  = FIGURES_DIR / "stage_2_regressor_importance.png"
            tab1, tab2 = st.tabs(["Stage 1 — Classifier", "Stage 2 — Regressor"])
            with tab1:
                if clf_img.exists():
                    st.image(str(clf_img), use_container_width=True)
                else:
                    st.warning(f"Image not found: {clf_img}")
            with tab2:
                if reg_img.exists():
                    st.image(str(reg_img), use_container_width=True)
                else:
                    st.warning(f"Image not found: {reg_img}")