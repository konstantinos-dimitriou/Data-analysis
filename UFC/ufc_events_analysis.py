# Author: Konstantinos Dimitriou
# Date: 09/06/2025

# Note: The data sets have been made available by the user maksbasher and can be found on Kaggle at the following URL
# https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import TESTS.utility_function_tests as tests

PLOTS_DIR = "PLOTS"
os.makedirs(PLOTS_DIR, exist_ok=True)

WEIGHT_DIVISIONS = {
        "male": [
            "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
            "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"
        ],
        "female": [
            "Women's Strawweight", "Women's Flyweight",
            "Women's Bantamweight", "Women's Featherweight"
        ]
    }


def check_data_completeness(df):
    data_entry_numbers = df.isnull().sum()
    data_holes = data_entry_numbers[data_entry_numbers !=0]
    if len(data_holes) != 0:
        print("Holes in data:")
        print(data_holes)
        return False
    else:
        return True


def normalize_weight_classes(x, weight_divisions=WEIGHT_DIVISIONS):
    # The order passed on from the WEIGHT_DIVISIONS dictionary in the normalisation here is crucial.
    # e.g. With this order Women's Bantamweight are not turned into Bantamweight.
    # Female weightclasses
    for weight_class in weight_divisions["female"]:
        if weight_class in x:
            return weight_class
    # Male weightclasses
    for weight_class in weight_divisions["male"]:
        if weight_class in x:
            return weight_class
    return x


def clean_weight_classes(data):
    """Remove non-standard entries and normalize weight class labels.
    Accepts either a Series (weight class in index) or a DataFrame (weight class as a column).
    """
    FILTER_PATTERN = "Ultimate|TUF|Catch|Superfight|Open Weight|Super Heavyweight"
    TOURNAMENT_PATTERN = r"^UFC \d+ Tournament Title$"

    if isinstance(data, pd.Series):
        data = data[~data.index.str.contains(FILTER_PATTERN)]
        data = data[~data.index.str.contains(TOURNAMENT_PATTERN)]
        data.index = data.index.map(normalize_weight_classes)
        data = data.groupby(data.index).sum()
    elif isinstance(data, pd.DataFrame):
        data = data[~data["weight_class"].str.contains(FILTER_PATTERN)]
        data = data[~data["weight_class"].str.contains(TOURNAMENT_PATTERN)]
        data["weight_class"] = data["weight_class"].map(normalize_weight_classes)
        data = data.groupby(["weight_class", "method"])["count"].sum().reset_index()

    return data


def build_weightclass_color_map(weightclass_data, weight_divisions=WEIGHT_DIVISIONS):
    WEIGHT_COLOR_PALETTES = {
        "male":   ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"],  # green → yellow → orange → deep red
        "female": ["#aed6f1", "#5dade2", "#a569bd", "#e91e8c"]   # light blue → blue → purple → pink
    }
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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights by Weight Class | Pie-chart.png")
    plt.show()


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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights by Weight Class | Bar-chart.png")
    plt.show()


def weight_class_analysis(df):
    weights = df["weight_class"].value_counts()
    # Clean data
    weights = clean_weight_classes(weights)
    # Sort again
    weights = weights.sort_values(ascending=False)

    # Plot data 
    # Build color map
    color_map = build_weightclass_color_map(weights)
    # Plot results as a pie chart
    weightclass_pie_chart(weights, color_map)
    # Plot as bar chart
    weightclass_bar_chart(weights, color_map)

def derive_ko_totals(weights_method_df):
    total_fights = weights_method_df.groupby("weight_class")["count"].sum().sort_values(ascending=False)
    ko_fights = weights_method_df[weights_method_df["method"] == "KO/TKO"].groupby("weight_class")["count"].sum()
    ko_fights = ko_fights.reindex(total_fights.index, fill_value=0)
    return total_fights, ko_fights


def weight_class_method_pie_chart(total_fights, ko_fights, color_map):
    # Build colour lists for inner and outer layers
    outer_colors = [color_map[wc] for wc in total_fights.index]
    inner_vals = []
    inner_colors = []
    for wc in total_fights.index:
        ko_count = ko_fights[wc]
        inner_vals.append(ko_count)
        inner_vals.append(total_fights[wc] - ko_count)
        inner_colors.append(darken_color(color_map[wc]))
        inner_colors.append(color_map[wc])

    size = 0.3
    fig, ax = plt.subplots(figsize=(11, 8), num="UFC Total Fights and KO-TKO by Weight Class | Pie-chart")
    outer_wedges, _ = ax.pie(total_fights.values, radius=1, colors=outer_colors,
                             wedgeprops=dict(width=size, edgecolor="w"))
    ax.pie(inner_vals, radius=1-size, colors=inner_colors,
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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights and KO-TKO by Weight Class | Pie-chart.png")
    plt.show()


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
        ax.bar(x_positions[i],                   total,    width=bar_width, color=color,               edgecolor="w")
        ax.bar(x_positions[i] + bar_width + gap, ko_count, width=bar_width, color=darken_color(color), edgecolor="w")

    ax.set_xticks(x_positions + bar_width / 2 + gap / 2)
    ax.set_xticklabels([])  # remove labels

    legend_handles = [mpatches.Patch(color=color_map[wc], label=wc) for wc in total_fights.index]
    legend_handles.append(mpatches.Patch(color=darken_color((0.5, 0.5, 0.5, 1.0)), label="KO-TKO"))
    ax.legend(handles=legend_handles, title="Weight Classes", bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.set_xlabel("Weight Classes")
    ax.set_ylabel("Number of Fights")
    ax.set_title("UFC Total Fights and KO-TKO by Weight Class")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights and KO-TKO by Weight Class | Bar-chart.png")
    plt.show()


def weight_class_knockout_analysis(df):
    weights_method_df = df[["weight_class","method"]].value_counts().reset_index()
    # Clean data
    # Clean Methods (Methods are already clean)
    #method_series = df["method"].value_counts()
    #print(method_series)
    # Clean Weight classes
    weights_method_df = clean_weight_classes(weights_method_df)

    # Derive total fights and knockouts for plot
    total_fights, ko_fights = derive_ko_totals(weights_method_df)

    # Plot in pie and bar chart the total fights in each weight class and the corresponding total knockouts in that weight class.
    color_map = build_weightclass_color_map(weights_method_df["weight_class"].value_counts())
    weight_class_method_pie_chart(total_fights, ko_fights, color_map)
    weight_class_method_bar_chart(total_fights, ko_fights, color_map)


def main():
    # Set option to display the whole set
    pd.set_option("display.max_rows", None)

    # Load data set
    #df = pd.read_csv("DATA/Small_set/completed_events_small.csv")
    dataframe = pd.read_csv("DATA/Large_set/large_dataset.csv")

    # Check data completeness
    #check_data_completeness(dataframe)
    # Data is has a few holes regarding fighter age, reach and stance.

    # Check data structure
    #dataframe.info()
    #print(dataframe.head())

    # Run tests
    tests.test_weight_class_normalization()
    tests.test_clean_weight_classes()
    tests.test_derive_ko_totals()
    tests.test_build_weightclass_color_map()
    tests.test_darken_color()

    # Weight class analysis
    weight_class_analysis(dataframe)
    # Knockout analysis by weight
    weight_class_knockout_analysis(dataframe)





if __name__ == "__main__":
    main()