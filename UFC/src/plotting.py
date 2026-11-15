import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from .transform import WEIGHT_DIVISIONS


WEIGHT_COLOR_PALETTES = {
        "male":   ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"],  # green → yellow → orange → deep red
        "female": ["#aed6f1", "#5dade2", "#a569bd", "#e91e8c"]   # light blue → blue → purple → pink
    }


def build_weightclass_color_map(weightclass_data, weight_divisions=WEIGHT_DIVISIONS):
    present = set(weightclass_data.index)
    color_map = {}
    for gender in ("male", "female"):
        ordered = [c for c in weight_divisions[gender] if c in present]
        cmap = LinearSegmentedColormap.from_list(f"{gender}_palette", WEIGHT_COLOR_PALETTES[gender])
        colors = cmap(np.linspace(0, 1, len(ordered)))
        for c, col in zip(ordered, colors):
            color_map[c] = col
    return color_map


def darken_color(color, factor=0.6):
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


def weightclass_pie_chart(weightclass_data, color_map):
    colors = [color_map[c] for c in weightclass_data.index]
    fig, ax = plt.subplots(figsize=(11, 8), num="UFC Total Fights by Weight Class | Pie-chart")
    wedges, texts, autotexts = ax.pie(
        weightclass_data.values,
        labels=None,          # no labels on slices
        autopct='%1.1f%%',
        colors=colors
    )
    ax.legend(
        wedges,
        weightclass_data.index,
        title="Weight Classes",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )
    ax.set_title("UFC Total Fights by Weight Class")
    return fig


def weightclass_bar_chart(weightclass_data, color_map):
    weightclass_data = weightclass_data.sort_values(ascending=False)
    colors = [color_map[c] for c in weightclass_data.index]
    legend_handles = [mpatches.Patch(color=color_map[wc], label=wc) for wc in weightclass_data.index]

    fig, ax = plt.subplots(figsize=(12, 6), num="UFC Total Fights by Weight Class | Bar-chart")
    weightclass_data.plot.bar(ax=ax, color=colors)
    ax.set_xlabel("Weight Classes")
    ax.set_ylabel("Number of Fights")
    ax.set_title("UFC Total Fights by Weight Class")
    ax.set_xticks(range(len(weightclass_data)))
    ax.set_xticklabels([])
    ax.legend(handles=legend_handles, title="Weight Classes", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    return fig


def weight_class_method_pie_chart(total_fights, ko_fights, color_map):
    # Build colour lists for inner and outer layers
    outer_colors = [color_map[wc] for wc in total_fights.index]
    inner_vals = []
    inner_colors = []
    inner_labels = ()
    for wc in total_fights.index:
        ko_count = ko_fights[wc]
        
        inner_vals.append(total_fights[wc] - ko_count)
        inner_vals.append(ko_count)
        
        inner_colors.append(color_map[wc])
        inner_colors.append(darken_color(color_map[wc]))

        inner_labels += ("",)
        inner_labels += (f"{int(100*round(ko_count/total_fights[wc],2))}%",)

    size = 0.3
    fig, ax = plt.subplots(figsize=(11, 8), num="UFC Total Fights and KO-TKO by Weight Class | Pie-chart")
    outer_wedges, _ = ax.pie(total_fights.values, radius=1, colors=outer_colors,
                             wedgeprops=dict(width=size, edgecolor="w"))
    
    ax.pie(inner_vals, radius=1-size, colors=inner_colors, 
           labels=inner_labels,
           wedgeprops=dict(edgecolor="w"))

    # Build legend, one entry per weight class (outer colour) plus a KO indicator
    legend_handles = list(outer_wedges)
    legend_labels = list(total_fights.index)

    # Add a single KO/TKO example entry using a patch
    ko_patch = mpatches.Patch(color=darken_color((0.5, 0.5, 0.5, 1.0)), label="KO-TKO (darker shade)")
    legend_handles.append(ko_patch)
    legend_labels.append("KO/TKO")
    ax.legend(legend_handles, legend_labels,
              title="Weight Classes",
              loc="center left",
              bbox_to_anchor=(1, 0.5))
    ax.set(aspect="equal", title="UFC Total Fights and KO-TKO by Weight Class")
    plt.tight_layout()
    return fig


def weight_class_method_bar_chart(total_fights, ko_fights, color_map):
    # Set bar chart attributes
    n = len(total_fights)
    bar_width = 1.8
    gap = 0.01
    group_spacing = 0.3
    x_positions = np.arange(n) * (2 * bar_width + gap + group_spacing)

    fig, ax = plt.subplots(figsize=(16, 6), num="UFC Total Fights and KO-TKO by Weight Class | Bar-chart")
    for i, wc in enumerate(total_fights.index):
        color = color_map[wc]
        ko_count = ko_fights[wc]
        total = total_fights[wc]
        total_bar = ax.bar(x_positions[i], total, width=bar_width, label="", color=color, edgecolor="w")
        ko_bar = ax.bar(x_positions[i] + bar_width + gap, ko_count, width=bar_width, label=f"{round(ko_count/total,2)}", color=darken_color(color), edgecolor="w")
        ax.bar_label(ko_bar, fmt=f"{int(100*round(ko_count/total,2))}%%")

    ax.set_xticks(x_positions + bar_width / 2 + gap / 2)
    ax.set_xticklabels([])  # remove labels

    legend_handles = [mpatches.Patch(color=color_map[wc], label=wc) for wc in total_fights.index]
    legend_handles.append(mpatches.Patch(color=darken_color((0.5, 0.5, 0.5, 1.0)), label="KO-TKO"))
    ax.legend(handles=legend_handles, title="Weight Classes", bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.set_xlabel("Weight Classes")
    ax.set_ylabel("Number of Fights")
    ax.set_title("UFC Total Fights and KO-TKO by Weight Class")
    plt.tight_layout()
    return fig


def plot_winner_by_colour(red_wins, blue_wins, draws):
    plot_data = [red_wins, blue_wins, draws]
    labels = ["Red", "Blue", "Draw"]
    colours = ["#c21d1d", "#263ec7", "#cfa915"]
    legend_labels = [f"Red wins: {red_wins}", f"Blue wins: {blue_wins}", f"Draws: {draws}"]
    total_fights = sum(plot_data)
    fig, ax = plt.subplots(figsize=(11, 8), num="UFC fight result by corner colour-Pie-chart")
    wedges, texts, autotexts = ax.pie(
        plot_data,
        labels=labels,
        autopct='%1.1f%%',
        colors=colours
    )
    ax.legend(
        wedges,
        legend_labels,
        title=f"Total fight results (#{total_fights})",
        loc="lower left",
        bbox_to_anchor=(1, 0.5)
    )
    ax.set_title("UFC fight result by corner colour")
    plt.tight_layout()
    return fig


def plot_weeks_off_bar_chart(weeks_off_counts, tick_step=5):
    fig, ax = plt.subplots(figsize=(14, 6), num="UFC Weeks Off Between Fights-Bar-chart")
    weeks_off_counts.plot.bar(ax=ax, color="#3498db")

    # Only label every Nth tick to keep the x-axis readable given the long tail
    ax.set_xticks(ax.get_xticks()[::tick_step])

    ax.set_xlabel("Weeks Off Since Previous Fight")
    ax.set_ylabel("Number of Fights")
    ax.set_title("Distribution of Time Off Between UFC Fights")
    plt.tight_layout()
    return fig


def plot_winrate_by_layoff_period_line(layoff_period_winrates):
    fig, ax = plt.subplots(figsize=(10, 6), num="UFC Win Rate by Layoff Period-Line-Plot")
    x_labels = list(layoff_period_winrates.index.astype(str))
    y_values = layoff_period_winrates.values * 100
    reference_y_values = 50 * layoff_period_winrates.values/layoff_period_winrates.values

    ax.plot(x_labels, y_values, marker="o", color="#3498db")
    ax.plot(x_labels, reference_y_values, linestyle='dashed', color="#000000")
    ax.set_xlabel("Layoff Period (in days)")
    ax.set_ylabel("Win Percentage")
    ax.set_title("Average Win Percentage by Layoff Period")
    ax.set_ylim(30, 70)
    plt.tight_layout()
    return fig


def plot_matchup_winrate_heatmap(matchup_table):
    fig, ax = plt.subplots(figsize=(9, 8), num="UFC Win Rate by Layoff Matchup-Heatmap")
    data = matchup_table.values * 100

    im = ax.imshow(data, cmap="YlGn", vmin=20, vmax=80, origin="lower")

    ax.set_xticks(range(len(matchup_table.columns)))
    ax.set_xticklabels(matchup_table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matchup_table.index)))
    ax.set_yticklabels(matchup_table.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="black")

    ax.set_xlabel("Opponent's Layoff Period")
    ax.set_ylabel("This Fighter's Layoff Period")
    ax.set_title("Matchup Win Percentages by Layoff Period (in days) \n(row = this fighter, column = opponent)")
    fig.colorbar(im, ax=ax, label="Win %")
    plt.tight_layout()
    return fig


def save_figure(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)

