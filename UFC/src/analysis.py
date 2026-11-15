import numpy as np
import pandas as pd


def derive_ko_totals(weights_method_df):
    total_fights = weights_method_df.groupby("weight_class")["count"].sum().sort_values(ascending=False)
    ko_fights = weights_method_df[weights_method_df["method"] == "KO/TKO"].groupby("weight_class")["count"].sum()
    ko_fights = ko_fights.reindex(total_fights.index, fill_value=0)
    return total_fights, ko_fights


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


def compute_overall_winrate_by_layoff_period(long_df):
    return long_df.groupby("own_layoff_period", observed=False)["won"].mean()


def compute_matchup_winrate_table(long_df):
    return long_df.groupby(["own_layoff_period", "opponent_layoff_period"], observed=False)["won"].mean().unstack("opponent_layoff_period")


def compute_weight_class_counts(df):
    return df["weight_class"].value_counts()

