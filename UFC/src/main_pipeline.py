from pathlib import Path
import pandas as pd

from . import extract, transform, load, analysis, plotting

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

LARGE_CSV = RAW_DIR / "Large_set" / "large_dataset.csv"
MEDIUM_CSV = RAW_DIR / "Medium_set" / "medium_dataset.csv"

CORNER_CUTOFF_DATE = "3/21/2010"


def run_weight_class(large_df):
    counts = analysis.compute_weight_class_counts(large_df)
    counts = transform.clean_weight_classes(counts).sort_values(ascending=False)
    color_map = plotting.build_weightclass_color_map(counts)

    fig = plotting.weightclass_pie_chart(counts, color_map)
    plotting.save_figure(fig, "weight_class_pie", FIGURES_DIR)
    fig = plotting.weightclass_bar_chart(counts, color_map)
    plotting.save_figure(fig, "weight_class_bar", FIGURES_DIR)


def run_weight_class_knockouts(large_df):
    wm_df = large_df[["weight_class", "method"]].value_counts().reset_index()
    wm_df = transform.clean_weight_classes(wm_df)
    total_fights, ko_fights = analysis.derive_ko_totals(wm_df)
    color_map = plotting.build_weightclass_color_map(wm_df["weight_class"].value_counts())

    fig = plotting.weight_class_method_pie_chart(total_fights, ko_fights, color_map)
    plotting.save_figure(fig, "weight_class_ko_pie", FIGURES_DIR)
    fig = plotting.weight_class_method_bar_chart(total_fights, ko_fights, color_map)
    plotting.save_figure(fig, "weight_class_ko_bar", FIGURES_DIR)


def run_corner_colour(fights_df):
    red, blue, draws = analysis.derive_corner_win_counts(fights_df, cutoff_date=CORNER_CUTOFF_DATE)
    fig = plotting.plot_winner_by_colour(red, blue, draws)
    plotting.save_figure(fig, "corner_colour_pie", FIGURES_DIR)


def run_ring_rust(fights_rest_df):
    weeks_off = analysis.compute_weeks_off_distribution(fights_rest_df, winners_only=True)
    fig = plotting.plot_weeks_off_bar_chart(weeks_off)
    plotting.save_figure(fig, "weeks_off_bar", FIGURES_DIR)

    long_df = transform.compute_layoff_period_matchups(fights_rest_df)
    overall = analysis.compute_overall_winrate_by_layoff_period(long_df)
    fig = plotting.plot_winrate_by_layoff_period_line(overall)
    plotting.save_figure(fig, "winrate_by_layoff_line", FIGURES_DIR)

    matchup = analysis.compute_matchup_winrate_table(long_df)
    fig = plotting.plot_matchup_winrate_heatmap(matchup)
    plotting.save_figure(fig, "winrate_by_layoff_heatmap", FIGURES_DIR)


def main():
    pd.set_option("display.max_rows", None)

    # Extract
    large_df = extract.read_large_dataset(LARGE_CSV)
    medium_df = extract.read_medium_dataset(MEDIUM_CSV)

    # Transform (the merge is done once and shared)
    fights_df = transform.merge_large_and_medium_data(large_df, medium_df)
    fights_rest_df = transform.add_rest_days_columns(fights_df)
    fights_rest_df = transform.exclude_same_night_repeat_fights(fights_rest_df)

    # Load (the tables that were actually used)
    load.save_processed(fights_df, "fights_merged", PROCESSED_DIR)
    load.save_processed(fights_rest_df, "fights_with_rest_days", PROCESSED_DIR)

    # Analysis + plots
    run_weight_class(large_df)
    run_weight_class_knockouts(large_df)
    run_corner_colour(fights_df)
    run_ring_rust(fights_rest_df)


if __name__ == "__main__":
    main()

