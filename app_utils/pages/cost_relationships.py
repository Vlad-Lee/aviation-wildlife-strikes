import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app_utils.aggregations import compute_correlation, get_scatter_data
from app_utils.filters import apply_filters, get_sidebar_filters
from app_utils.plotting import show_plot, style_chart, style_legend


def show_scatter_plot(
    df_states,
    min_year: int,
    max_year: int,
    operator_list: list[str],
    time_of_day_list: list[str],
    aircraft_list: list[str],
) -> None:
    st.title("Do Flight Variables Explain Cost?")

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

    with st.sidebar:
        x_variable = st.selectbox(
            "Select X Variable",
            options=["Height", "Speed"],
            index=0,
        )

        warned_options = ["No", "Unknown", "Yes"]

        selected_warned = st.multiselect(
            "Filter Warned Flag",
            options=warned_options,
            default=warned_options,
        )

    st.caption(
        "Log cost is compared against flight variables to reduce the influence of extreme values. "
        f"Year range: {start_year} to {end_year}"
    )

    x_column_map = {
        "Height": "HEIGHT",
        "Speed": "SPEED",
    }
    x_col = x_column_map[x_variable]

    filtered_df = apply_filters(
        df_states=df_states,
        start_year=start_year,
        end_year=end_year,
        operator=selected_operator,
        time_of_day=selected_time_of_day,
        aircraft=selected_aircraft,
    )

    scatter_df = get_scatter_data(filtered_df, x_col)

    # Apply warned filter
    scatter_df = scatter_df[scatter_df["WARNED"].isin(selected_warned)]

    if scatter_df.empty:
        st.warning("No records match the selected filters.")
        return

    pearson_r = compute_correlation(scatter_df, x_col)

    st.metric(
        label=f"Pearson R: Cost vs. {x_variable}",
        value=f"{pearson_r:.2f}",
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = scatter_df["WARNED"].nunique()
    use_color = n_groups > 1

    if use_color:
        plt.subplots_adjust(right=0.8)

        sns.scatterplot(
            data=scatter_df,
            x=f"Log_{x_col}",
            y="Log_Cost",
            hue="WARNED",
            hue_order=["No", "Unknown", "Yes"],
            ax=ax,
            alpha=0.7,
        )
    else:
        sns.scatterplot(
            data=scatter_df,
            x=f"Log_{x_col}",
            y="Log_Cost",
            ax=ax,
            alpha=0.45,
            legend=False,
        )

    ax.set_xlabel(f"Log {x_variable}")
    ax.set_ylabel("Log Cost")
    ax.set_title(f"Log Cost vs. Log {x_variable}", color="white")

    style_chart(ax)

    # Legend styling + positioning
    legend = ax.get_legend()
    if legend:
        legend.set_title("Warned")
        style_legend(legend)

    show_plot(fig)

    st.caption(
        "Relationships are weak, but directionally meaningful. Costs increase slightly with speed "
        "and decrease with height, suggesting that the most severe strikes occur closer to the ground "
        "during critical flight phases."
    )