import matplotlib.pyplot as plt
import streamlit as st


def show_plot(fig) -> None:
    """Render and close a matplotlib figure."""
    st.pyplot(fig)
    plt.close(fig)


def style_chart(ax) -> None:
    """Apply dashboard dark theme styling to a matplotlib axis."""
    ax.set_facecolor("#151c26")
    ax.figure.set_facecolor("#151c26")

    ax.tick_params(axis="x", colors="#d1d5db")
    ax.tick_params(axis="y", colors="#d1d5db")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")

    ax.grid(axis="y", linestyle="--", alpha=0.2)


def style_legend(legend) -> None:
    """Style and position a matplotlib legend for the dark theme."""
    if not legend:
        return

    legend.set_bbox_to_anchor((1.02, 1))
    legend.set_loc("upper left")

    legend.get_frame().set_facecolor("#151c26")
    legend.get_frame().set_edgecolor("#6b7280")
    legend.get_frame().set_alpha(0.9)

    for text in legend.get_texts():
        text.set_color("white")

    legend.get_title().set_color("white")