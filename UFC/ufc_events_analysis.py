# Author: Konstantinos Dimitriou
# Date: 09/06/2025

# Note: The data sets have been made available by the user 'maksbasher' and can be found on Kaggle at the following URL
# https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024

# The "ring rust" analysis was done by the user 'Purple Belt Analytics' on YouTube. We repeat the same analysis here.
# https://www.youtube.com/watch?v=D3kaBoUZAuA

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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights by Weight Class-Pie-chart.png")
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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights by Weight Class-Bar-chart.png")
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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights and KO-TKO by Weight Class-Pie-chart.png")
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
    plt.savefig(f"{PLOTS_DIR}/UFC Total Fights and KO-TKO by Weight Class-Bar-chart.png")
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


def extract_event_dates(medium_df):
    """Return a dictionary mapping each event to its unique date.

    Raises
    ------
    ValueError
        If an event is associated with more than one date.
    """
    event_dates = medium_df.groupby("event")["date"].nunique()

    if (event_dates > 1).any():
        bad_events = event_dates[event_dates > 1].index.tolist()
        raise ValueError(
            f"The following events have multiple dates: {bad_events}"
        )

    return (
        medium_df[["event", "date"]]
        .drop_duplicates()
        .set_index("event")["date"]
        .to_dict()
    )


def merge_large_and_medium_data(large_df, medium_df):
    """Combine the decisive fights from large_df with the draws from medium_df.

    The returned dataframe has columns
        event_name, r_fig   hter, b_fighter, date, winner
    where winner is one of {"Red", "Blue", "draw"}.
    """
    # This functions changes the fight order, inside an event.

    # Extract relevant columns from data frames.
    medium_df = medium_df[["event", "r_fighter", "b_fighter", "date", "status"]]
    large_df = large_df[["event_name", "r_fighter", "b_fighter", "winner"]]

    # Extract event dates.
    event_date_dict = extract_event_dates(medium_df)  

    # Add the date to the large data frame.
    large_df = large_df.copy() # Create a copy here since we will add a column
    large_df["date"] = large_df["event_name"].map(event_date_dict)

    # Remove events with missing dates.
    missing = large_df.loc[large_df["date"].isna(), "event_name"].unique()
    if len(missing) > 0:
        print("Removing events with missing dates:", missing)

    large_df = large_df.dropna(subset=["date"]).reset_index(drop=True)

    # Keep only the draws from the medium dataset (wins have mixed up the fighters).
    medium_df = medium_df[medium_df["status"] == "draw"].copy() # This also drops the fights with status="Fight was not properly finished".
    medium_df["status"] = "Draw"  # Normalize casing to match "Red"/"Blue".

    # Rename event,status to match large_df.
    medium_df = medium_df.rename(columns={"event": "event_name"})
    medium_df = medium_df.rename(columns={"status": "winner"})

    # Match column order.
    medium_df = medium_df[["event_name", "r_fighter", "b_fighter", "winner", "date"]]
    large_df = large_df[["event_name", "r_fighter", "b_fighter", "winner", "date"]]

    # Concatenate the two datasets.
    df = pd.concat([large_df, medium_df], ignore_index=True)

    # Sort chronologically by date using a stable sort, so that fights sharing the
    # same date (every fight within a single event) retain their original relative
    # order from the source data — which reflects the true chronological order fights
    # occurred within that event (important for early tournament-style events where
    # a fighter could face multiple opponents on the same night).
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    #df = df.sort_values("date", ascending=False).reset_index(drop=True)
    df = df.sort_values("date", ascending=False, kind="stable").reset_index(drop=True) # We keep the order of fights. This is important when exluding succesive tournament fights.
    df["date"] = df["date"].dt.strftime("%m/%d/%Y")

    return df


def derive_corner_win_counts(merged_df, cutoff_date=None):
    """Count fights won by red corner, blue corner, and draws.
    cutoff_date: only include fights on or after this date (string, format mm/dd/yyyy).
                 None means no cutoff is applied.
    """
    dates = pd.to_datetime(merged_df["date"], format="%m/%d/%Y")

    if cutoff_date is not None:
        cutoff = pd.to_datetime(cutoff_date, format="%m/%d/%Y")
        merged_df = merged_df[dates >= cutoff]

    red_wins  = merged_df[merged_df["winner"] == "Red"].shape[0]

    blue_wins = merged_df[merged_df["winner"] == "Blue"].shape[0]
    
    draws     = merged_df[merged_df["winner"] == "Draw"].shape[0]

    return red_wins, blue_wins, draws


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
    plt.savefig(f"{PLOTS_DIR}/UFC fight result by corner colour-Pie-chart.png")
    plt.show()


def corner_colour_analysis(large_df, medium_df):
    CUTOFF_DATE = "3/21/2010"
    merged_df = merge_large_and_medium_data(large_df, medium_df)
    red_wins, blue_wins, draws = derive_corner_win_counts(merged_df, cutoff_date=CUTOFF_DATE)
    plot_winner_by_colour(red_wins, blue_wins, draws)


def add_rest_days_columns(df):
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # Long format: one row per (fight, corner_colour)
    red = df[["r_fighter", "date_parsed"]].rename(columns={"r_fighter": "fighter"})
    red["corner"] = "Red"
    red["row_index"] = df.index

    blue = df[["b_fighter", "date_parsed"]].rename(columns={"b_fighter": "fighter"})
    blue["corner"] = "Blue"
    blue["row_index"] = df.index

    long_df = pd.concat([red, blue], ignore_index=True)

    # Sort by fighter, then date ascending, so the previous fight sits directly above
    long_df = long_df.sort_values(["fighter", "date_parsed"])

    # Days since the fighter's previous fight; NaN for a fighter's first fight
    long_df["days_since_last_fight"] = long_df.groupby("fighter")["date_parsed"].diff().dt.days

    # Map back onto the original dataframe via row_index and corner
    red_days = long_df[long_df["corner"] == "Red"].set_index("row_index")["days_since_last_fight"]
    blue_days = long_df[long_df["corner"] == "Blue"].set_index("row_index")["days_since_last_fight"]

    df["r_rest_days"] = red_days
    df["b_rest_days"] = blue_days

    df = df.drop(columns=["date_parsed"])
    return df


def exclude_same_night_repeat_fights(df):
    """For fighters who fought multiple times on the same date (early tournament-era
    events), only the first fight of the night carries meaningful rest-day information.
    Any later same-night fight has 0 rest days purely as a structural artifact of the
    tournament format, not because of anything informative about the fighter. This
    function nulls out rest_days for those later same-night appearances, treating them
    the same as a fighter's true first fight (no prior information available).
    """
    df = df.copy()

    # Build long format to identify, per fighter, which rows are same-night repeats
    long_df = pd.concat([
        df[["r_fighter", "date"]].rename(columns={"r_fighter": "fighter"}).assign(corner="Red", row_index=df.index),
        df[["b_fighter", "date"]].rename(columns={"b_fighter": "fighter"}).assign(corner="Blue", row_index=df.index),
    ])

    # Rank each fighter's fights on a given date by their original row order
    # (row order reflects chronological order within the event, per fight numbering)
    long_df["same_night_rank"] = long_df.groupby(["fighter", "date"]).cumcount()

    # Anything with rank > 0 is a same-night repeat — not their first fight that night
    repeats = long_df[long_df["same_night_rank"] > 0]

    red_repeat_rows = repeats.loc[repeats["corner"] == "Red", "row_index"]
    blue_repeat_rows = repeats.loc[repeats["corner"] == "Blue", "row_index"]

    df.loc[red_repeat_rows, "r_rest_days"] = np.nan
    df.loc[blue_repeat_rows, "b_rest_days"] = np.nan

    return df


def compute_weeks_off_distribution(df, winners_only=False):
    """Combine r_rest_days and b_rest_days into one series, convert to weeks,
    drop fighters' first fights (NaN), and count fights per number of weeks off.

    winners_only: if True, only count the rest days of the fighter who won that fight.
                  Draws are naturally excluded since neither corner is "the winner".
    """
    if winners_only:
        red_days = df.loc[df["winner"] == "Red", "r_rest_days"]
        blue_days = df.loc[df["winner"] == "Blue", "b_rest_days"]
        rest_days = pd.concat([red_days, blue_days], ignore_index=True)
    else:
        rest_days = pd.concat([df["r_rest_days"], df["b_rest_days"]], ignore_index=True)

    rest_days = rest_days.dropna()
    weeks_off = (rest_days // 7).astype(int)
    weeks_off_counts = weeks_off.value_counts().sort_index()
    return weeks_off_counts


def plot_weeks_off_bar_chart(weeks_off_counts, tick_step=5):
    fig, ax = plt.subplots(figsize=(14, 6), num="UFC Weeks Off Between Fights-Bar-chart")
    weeks_off_counts.plot.bar(ax=ax, color="#3498db")

    # Only label every Nth tick to keep the x-axis readable given the long tail
    ax.set_xticks(ax.get_xticks()[::tick_step])

    ax.set_xlabel("Weeks Off Since Previous Fight")
    ax.set_ylabel("Number of Fights")
    ax.set_title("Distribution of Time Off Between UFC Fights")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/UFC Weeks Off Between Fights-Bar-chart.png")
    plt.show()


LAYOFF_PERIOD_BINS = [-0.1, 90, 180, 270, 360, 540, np.inf]
LAYOFF_PERIOD_LABELS = ["0-90", "91-180", "181-270", "271-360", "361-540", "540+"]


def categorize_rest_days(rest_days):
    """Bucket rest days into layoff periods. NaN stays NaN (no prior fight)."""
    return pd.cut(rest_days, bins=LAYOFF_PERIOD_BINS, labels=LAYOFF_PERIOD_LABELS, ordered=True)


def compute_layoff_period_matchups(df):
    """Build a long-format table: one row per fighter per fight, with their own
    layoff period, their opponent's layoff period, and whether they won.

    Only decisive fights (Red/Blue winner) are included — draws don't have a
    meaningful "win percentage" contribution. Fights where either corner's
    layoff period is undefined (a fighter's first fight) are excluded.
    """
    # Drop irrelevant data.
    df = df.dropna(subset=["r_rest_days", "b_rest_days"]).copy() # Drop NaN in fighter rest days i.e. fighters which fight for the first time.
    df = df[df["winner"].isin(["Red", "Blue"])] # Drop fights which ended in a draw.

    # Catergorize rest days into discete groups.
    df["r_layoff_period"] = categorize_rest_days(df["r_rest_days"])
    df["b_layoff_period"] = categorize_rest_days(df["b_rest_days"])

    # Build long format table
    red_rows = pd.DataFrame({
        "own_layoff_period": df["r_layoff_period"],
        "opponent_layoff_period": df["b_layoff_period"],
        "won": (df["winner"] == "Red").astype(int)
    })
    blue_rows = pd.DataFrame({
        "own_layoff_period": df["b_layoff_period"],
        "opponent_layoff_period": df["r_layoff_period"],
        "won": (df["winner"] == "Blue").astype(int)
    })

    return pd.concat([red_rows, blue_rows], ignore_index=True)


def compute_overall_winrate_by_layoff_period(long_df):
    return long_df.groupby("own_layoff_period", observed=False)["won"].mean()


def compute_matchup_winrate_table(long_df):
    return long_df.groupby(["own_layoff_period", "opponent_layoff_period"], observed=False)["won"].mean().unstack("opponent_layoff_period")


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
    plt.savefig(f"{PLOTS_DIR}/UFC Win Rate by Layoff Period-Line-Plot.png")
    plt.show()



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
    plt.savefig(f"{PLOTS_DIR}/UFC Matchup Win Rate by Layoff-Heatmap.png")
    plt.show()


def win_rate_by_layoff_analysis(df):
    long_df = compute_layoff_period_matchups(df)

    overall_winrates = compute_overall_winrate_by_layoff_period(long_df)
    plot_winrate_by_layoff_period_line(overall_winrates)

    matchup_table = compute_matchup_winrate_table(long_df)
    plot_matchup_winrate_heatmap(matchup_table)


def ring_rust_analysis(large_df, medium_df):
    df = merge_large_and_medium_data(large_df, medium_df) # Has test
    df = add_rest_days_columns(df) # Has test

    df = exclude_same_night_repeat_fights(df)

    # Drop irrelevant columns
    df = df.drop(["event_name",
             "r_fighter",
             "b_fighter",
             "date"
             ],
             axis=1)
    
    # Compute and plot weeks of 
    weeks_off_counts = compute_weeks_off_distribution(df, winners_only=True)
    plot_weeks_off_bar_chart(weeks_off_counts)

    # Compute win rates by layoff time
    win_rate_by_layoff_analysis(df)


def main():
    # Set option to display the whole set
    pd.set_option("display.max_rows", None)

    # Load data set
    #df = pd.read_csv("DATA/Small_set/completed_events_small.csv")
    large_df = pd.read_csv("DATA/Large_set/large_dataset.csv")  # Does not contain the date
    medium_df = pd.read_csv("DATA/Medium_set/medium_dataset.csv") # Is needed for the date events, but does not contain the winner.

    # Check data completeness
    #check_data_completeness(large_df)
    #check_data_completeness(medium_df)
    # See Notes_on_data_sets.txt
    # We remove in the medium set, the last 85 lines consisting of ,,,,,Fight was not properly finished,,,,,,,,,,,,, .

    # Check data structure
    #large_df.info()
    #print(large_dataframe.head())

    # Run tests
    RUN_TESTS = True
    #RUN_TESTS = False
    if RUN_TESTS:
        # Weight and KO functions
        tests.test_weight_class_normalization()
        tests.test_clean_weight_classes()
        tests.test_derive_ko_totals()
        tests.test_build_weightclass_color_map()
        tests.test_darken_color()
        # Cornern colour functions
        tests.test_extract_event_dates()
        tests.test_cutoff_excludes_all()
        tests.test_merge_large_and_medium_data()
        tests.test_derive_corner_win_counts()
        # Ring rust functions
        tests.test_add_rest_days_columns()
        tests.test_compute_weeks_off_distribution()
        tests.test_exclude_same_night_repeat_fights()
        tests.test_categorize_rest_days()
        tests.test_compute_layoff_period_matchups()
        tests.test_compute_overall_winrate_by_layoff_period()
        tests.test_compute_matchup_winrate_table()

    # Weight class analysis
    weight_class_analysis(large_df)
    # Knockout analysis by weight
    weight_class_knockout_analysis(large_df)

    # Corner colour impact
    # Both dataframes are needed since large does not contain draws and medium does not contain the winning side
    corner_colour_analysis(large_df, medium_df)

    # Ring rust analysis
    ring_rust_analysis(large_df, medium_df)


    # Other questions:
    # Which referees have the most knockouts?
    # Does the younger fighter usually win?
    

if __name__ == "__main__":
    main()