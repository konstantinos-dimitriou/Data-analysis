# Write tests for ring rust.

import pandas as pd
import numpy as np
from ufc_events_analysis import (
    normalize_weight_classes,
    clean_weight_classes,
    derive_ko_totals,
    build_weightclass_color_map,
    darken_color,
    WEIGHT_DIVISIONS,
    extract_event_dates,
    derive_corner_win_counts,
    merge_large_and_medium_data,
    add_rest_days_columns,
    compute_weeks_off_distribution,
    exclude_same_night_repeat_fights,
    categorize_rest_days,
    compute_layoff_period_matchups,
    LAYOFF_PERIOD_LABELS,
    compute_overall_winrate_by_layoff_period,
    compute_matchup_winrate_table
)


def test_weight_class_normalization():
    cases = {
        # Base classes (must stay unchanged)
        "Bantamweight": "Bantamweight",
        "Featherweight": "Featherweight",
        "Flyweight": "Flyweight",
        "Heavyweight": "Heavyweight",
        "Light Heavyweight": "Light Heavyweight",
        "Lightweight": "Lightweight",
        "Middleweight": "Middleweight",
        "Welterweight": "Welterweight",

        # UFC titles
        "UFC Bantamweight Title": "Bantamweight",
        "UFC Featherweight Title": "Featherweight",
        "UFC Flyweight Title": "Flyweight",
        "UFC Heavyweight Title": "Heavyweight",
        "UFC Light Heavyweight Title": "Light Heavyweight",
        "UFC Lightweight Title": "Lightweight",
        "UFC Middleweight Title": "Middleweight",
        "UFC Welterweight Title": "Welterweight",

        # Interim titles
        "UFC Interim Bantamweight Title": "Bantamweight",
        "UFC Interim Featherweight Title": "Featherweight",
        "UFC Interim Flyweight Title": "Flyweight",
        "UFC Interim Heavyweight Title": "Heavyweight",
        "UFC Interim Light Heavyweight Title": "Light Heavyweight",
        "UFC Interim Lightweight Title": "Lightweight",
        "UFC Interim Middleweight Title": "Middleweight",
        "UFC Interim Welterweight Title": "Welterweight",

        # Women’s divisions
        "Women's Bantamweight": "Women's Bantamweight",
        "Women's Featherweight": "Women's Featherweight",
        "Women's Flyweight": "Women's Flyweight",
        "Women's Strawweight": "Women's Strawweight",

        "UFC Women's Bantamweight Title": "Women's Bantamweight",
        "UFC Women's Featherweight Title": "Women's Featherweight",
        "UFC Women's Flyweight Title": "Women's Flyweight",
        "UFC Women's Strawweight Title": "Women's Strawweight",

        # Tournament titles (edge cases)
        "UFC 13 Heavyweight Tournament Title": "Heavyweight",
        "UFC 13 Lightweight Tournament Title": "Lightweight",
        "UFC 14 Heavyweight Tournament Title": "Heavyweight",
        "UFC 14 Middleweight Tournament Title": "Middleweight",
        "UFC 15 Heavyweight Tournament Title": "Heavyweight",
        "UFC 17 Middleweight Tournament Title": "Middleweight",
    }

    for raw, expected in cases.items():
        result = normalize_weight_classes(raw,weight_divisions=WEIGHT_DIVISIONS)
        assert result == expected, f"Failed at: {raw} → {result}, expected {expected}."

    print("All weight_class_normalization tests passed.")


def test_clean_weight_classes():
    # Test Series path
    raw_series = pd.Series({
        "UFC Interim Lightweight Title":      10,
        "Lightweight":                         5,
        "TUF Lightweight":                     3,
        "UFC 13 Heavyweight Tournament Title": 2,
        "Open Weight":                         1,
    })
    cleaned_series = clean_weight_classes(raw_series)
    assert "TUF Lightweight"  not in cleaned_series.index, "TUF entry should be removed."
    assert "Open Weight"      not in cleaned_series.index, "Open Weight should be removed."
    assert "UFC 13 Heavyweight Tournament Title" not in cleaned_series.index, "Tournament title should be removed."
    assert cleaned_series["Lightweight"] == 15,            "Lightweight entries should be merged and summed."

    # Test DataFrame path
    raw_df = pd.DataFrame({
        "weight_class": ["UFC Interim Lightweight Title", "Lightweight", "TUF Lightweight", "Open Weight"],
        "method":       ["KO/TKO",                       "KO/TKO",      "KO/TKO",          "KO/TKO"],
        "count":        [10,                              5,             3,                  1],
    })
    cleaned_df = clean_weight_classes(raw_df)
    assert "TUF Lightweight" not in cleaned_df["weight_class"].values, "TUF entry should be removed."
    assert "Open Weight"     not in cleaned_df["weight_class"].values, "Open Weight should be removed."
    assert cleaned_df[cleaned_df["weight_class"] == "Lightweight"]["count"].values[0] == 15, \
        "Lightweight entries should be merged and summed."

    print("All clean_weight_classes tests passed.")


def test_derive_ko_totals():
    raw_df = pd.DataFrame({
        "weight_class": ["Lightweight", "Lightweight", "Lightweight", "Heavyweight", "Heavyweight"],
        "method":       ["KO/TKO",      "Submission",  "KO/TKO",      "KO/TKO",      "Decision - Unanimous"],
        "count":        [10,             5,             3,             8,             4],
    })

    total_fights, ko_fights = derive_ko_totals(raw_df)

    # Total fights should sum all methods per weight class
    assert total_fights["Lightweight"] == 18, f"Expected 18, got {total_fights['Lightweight']}"
    assert total_fights["Heavyweight"] == 12, f"Expected 12, got {total_fights['Heavyweight']}"

    # KO fights should only sum KO/TKO entries
    assert ko_fights["Lightweight"] == 13, f"Expected 13, got {ko_fights['Lightweight']}"
    assert ko_fights["Heavyweight"] == 8,  f"Expected 8, got {ko_fights['Heavyweight']}"

    # total_fights should be sorted descending
    assert list(total_fights.index) == ["Lightweight", "Heavyweight"], "total_fights should be sorted descending by fight count."

    # ko_fights should follow the same order as total_fights
    assert list(ko_fights.index) == list(total_fights.index), "ko_fights index order should match total_fights."

    # A weight class with no KO/TKO should return 0, not NaN
    raw_df_no_ko = pd.DataFrame({
        "weight_class": ["Flyweight"],
        "method":       ["Submission"],
        "count":        [5],
    })
    total_fights_no_ko, ko_fights_no_ko = derive_ko_totals(raw_df_no_ko)
    assert ko_fights_no_ko["Flyweight"] == 0, "Weight class with no KO/TKO should return 0, not NaN."

    print("All derive_ko_totals tests passed.")


def test_build_weightclass_color_map():
    # Test that all divisions get a color
    mock_data = pd.Series({wc: 1 for gender in WEIGHT_DIVISIONS.values() for wc in gender})
    color_map = build_weightclass_color_map(mock_data)
    all_divisions = WEIGHT_DIVISIONS["male"] + WEIGHT_DIVISIONS["female"]
    for wc in all_divisions:
        assert wc in color_map, f"Missing color for: {wc}"
    print("All weight_class_color_map tests passed.")

def test_darken_color():
    color = (1.0, 1.0, 1.0, 1.0)
    darkened = darken_color(color, factor=0.6)
    assert darkened == (0.6, 0.6, 0.6, 1.0), f"Unexpected darkened color: {darkened}"
    # Alpha should never be affected
    assert darkened[3] == color[3], "Alpha channel should not be darkened."
    print("All darken_color tests passed.")


def test_extract_event_dates():
    medium_df = pd.DataFrame({
        "event": ["Event A", "Event A", "Event B", "Event B", "Event C"],
        "date":  ["1/1/2020", "1/1/2020", "2/2/2020", "2/2/2020", "3/3/2020"]
    })

    result = extract_event_dates(medium_df)
    expected = {"Event A": "1/1/2020", "Event B": "2/2/2020", "Event C": "3/3/2020"}
    assert result == expected, f"Expected {expected}, got {result}"

    # Should raise if an event has inconsistent dates
    bad_df = pd.DataFrame({
        "event": ["Event A", "Event A"],
        "date":  ["1/1/2020", "1/2/2020"]
    })
    try:
        extract_event_dates(bad_df)
        assert False, "Expected ValueError for inconsistent event dates, but none was raised."
    except ValueError:
        pass

    print("All extract_event_dates tests passed.")


def test_derive_corner_win_counts():
    merged_df = pd.DataFrame({
        "date": [
            "01/01/2020",
            "01/01/2020",
            "02/01/2020",
            "03/01/2020",
        ],
        "winner": [
            "Red",
            "Blue",
            "Draw",
            "Red",
        ]
    })

    # --- no cutoff ---
    red, blue, draws = derive_corner_win_counts(merged_df)

    assert red == 2
    assert blue == 1
    assert draws == 1

    # --- cutoff excludes first two fights ---
    red, blue, draws = derive_corner_win_counts(
        merged_df,
        cutoff_date="01/15/2020"
    )

    # only last two rows remain:
    # Blue excluded, first Red excluded
    assert red == 1
    assert blue == 0
    assert draws == 1
    print("All derive_corner_win_counts tests passed.")



def test_merge_large_and_medium_data():
    # --- medium dataset (source of dates + draws) ---
    medium_df = pd.DataFrame({
        "event": ["UFC A", "UFC A", "UFC B"],
        "r_fighter": ["F1", "F2", "F3"],
        "b_fighter": ["F4", "F5", "F6"],
        "date": ["01/01/2020", "01/01/2020", "02/01/2020"],
        "status": ["win", "draw", "draw"],
    })

    # --- large dataset (decisive fights only) ---
    large_df = pd.DataFrame({
        "event_name": ["UFC A", "UFC B"],
        "r_fighter": ["F1", "F3"],
        "b_fighter": ["F4", "F6"],
        "winner": ["Red", "Blue"],
    })

    result = merge_large_and_medium_data(large_df, medium_df)

    # --- checks ---

    # 1. correct columns
    assert set(result.columns) == {
        "event_name", "r_fighter", "b_fighter", "winner", "date"
    }

    # 2. number of rows: 2 fights + 2 draws = 4 rows
    assert len(result) == 4

    # 3. draws are included
    assert (result["winner"] == "Draw").sum() == 2

    # 4. winners preserved correctly
    assert (result["winner"] == "Red").sum() == 1
    assert (result["winner"] == "Blue").sum() == 1

    # 5. date mapping works
    assert result["date"].notna().all()

    print("All merge_large_and_medium_data tests passed.")


def test_cutoff_excludes_all():
    df = pd.DataFrame({
        "date": ["01/01/2020"],
        "winner": ["Red"]
    })

    red, blue, draws = derive_corner_win_counts(
        df,
        cutoff_date="02/01/2020"
    )

    assert red == 0
    assert blue == 0
    assert draws == 0

    print("All cutoff_excludes_all tests passed.")


def test_add_rest_days_columns():
    df = pd.DataFrame({
        "r_fighter": ["Alice", "Bob",   "Alice"],
        "b_fighter": ["Carol", "Alice", "Dave"],
        "winner":    ["Red",   "Blue",  "Red"],
        "date":      ["3/1/2024", "2/1/2024", "1/1/2024"],
    })

    result = add_rest_days_columns(df)

    # Alice's fight on 3/1/2024 (red corner) — previous fight was 2/1/2024 (blue corner) = 29 days
    assert result.loc[0, "r_rest_days"] == 29, f"Expected 29, got {result.loc[0, 'r_rest_days']}"
    # Carol's only fight — no previous fight
    assert pd.isna(result.loc[0, "b_rest_days"]), "Carol's first fight should have NaN rest days."
    # Bob's only fight — no previous fight
    assert pd.isna(result.loc[1, "r_rest_days"]), "Bob's first fight should have NaN rest days."
    # Alice's fight on 2/1/2024 (blue corner) — previous fight was 1/1/2024 (red corner) = 31 days
    assert result.loc[1, "b_rest_days"] == 31, f"Expected 31, got {result.loc[1, 'b_rest_days']}"
    # Alice's fight on 1/1/2024 (red corner) is her earliest fight overall — no previous fight
    assert pd.isna(result.loc[2, "r_rest_days"]), "Alice's earliest fight should have NaN rest days."
    # Dave's only fight — no previous fight
    assert pd.isna(result.loc[2, "b_rest_days"]), "Dave's first fight should have NaN rest days."

    print("All add_rest_days_columns tests passed.")


def test_compute_weeks_off_distribution():
    df = pd.DataFrame({
        "winner":      ["Red", "Blue", "Red"],
        "r_rest_days": [14,    None,   21],
        "b_rest_days": [None,  35,     None],
    })

    # All fighters, NaN dropped
    result = compute_weeks_off_distribution(df, winners_only=False)
    assert result.get(2, 0) == 1, f"Expected 1 fight at 2 weeks off, got {result.get(2, 0)}"
    assert result.get(3, 0) == 1, f"Expected 1 fight at 3 weeks off, got {result.get(3, 0)}"
    assert result.get(5, 0) == 1, f"Expected 1 fight at 5 weeks off, got {result.get(5, 0)}"
    assert result.sum() == 3, "NaN values should not be counted at all, including as 0 weeks off."

    # Winners only — Red won fight 0 (14 days) and fight 2 (21 days); Blue won fight 1 (35 days)
    result_winners = compute_weeks_off_distribution(df, winners_only=True)
    assert result_winners.get(2, 0) == 1, f"Expected 1 winner fight at 2 weeks off, got {result_winners.get(2, 0)}"
    assert result_winners.get(3, 0) == 1, f"Expected 1 winner fight at 3 weeks off, got {result_winners.get(3, 0)}"
    assert result_winners.get(5, 0) == 1, f"Expected 1 winner fight at 5 weeks off, got {result_winners.get(5, 0)}"
    assert result_winners.sum() == 3, "Total winner fights should be 3, with no NaN miscounted as 0."

    print("All compute_weeks_off_distribution tests passed.")

def test_exclude_same_night_repeat_fights():
    df = pd.DataFrame({
        "date": ["01/01/2020", "01/01/2020"],
        "r_fighter": ["A", "A"],
        "b_fighter": ["B", "C"],
        "r_rest_days": [100, 0],
        "b_rest_days": [50, 60],
    })

    result = exclude_same_night_repeat_fights(df)

    assert result.loc[0, "r_rest_days"] == 100
    assert np.isnan(result.loc[1, "r_rest_days"]), "Second fight on same night should be NaN."
    assert result.loc[0, "b_rest_days"] == 50
    assert result.loc[1, "b_rest_days"] == 60

    print("All exclude_same_night_repeat_fights tests passed.")


def test_categorize_rest_days():
    rest_days = pd.Series([30, 120, 250, 330, 500, 700, np.nan])

    result_values = categorize_rest_days(rest_days)

    expected_values = ["0-90", "91-180", "181-270", "271-360", "361-540", "540+", np.nan]

    for result, expected in zip(result_values.astype(object), expected_values):
        if pd.isna(expected):
            assert pd.isna(result), "NaN should remain NaN."
        else:
            assert result == expected, f"Expected {expected}, got {result}"

    print("All categorize_rest_days tests passed.")


def test_compute_layoff_period_matchups():
    df = pd.DataFrame({
        "winner": ["Red", "Blue", "draw"],
        "r_rest_days": [30, 120, 30],
        "b_rest_days": [120, 30, 30],
    })

    result = compute_layoff_period_matchups(df)

    assert len(result) == 4, "Draw should be removed, leaving two fights -> four fighter rows."

    assert result["won"].sum() == 2, "Exactly two fighters should be winners."

    assert set(result["own_layoff_period"].astype(str)) == {"0-90", "91-180"}

    print("All compute_layoff_period_matchups tests passed.")


def test_compute_overall_winrate_by_layoff_period():
    long_df = pd.DataFrame({
        "own_layoff_period": pd.Categorical(
            ["0-90", "0-90", "91-180", "91-180"],
            categories=LAYOFF_PERIOD_LABELS,
            ordered=True,
        ),
        "won": [1, 0, 1, 1],
    })

    result = compute_overall_winrate_by_layoff_period(long_df)

    assert result.loc["0-90"] == 0.5
    assert result.loc["91-180"] == 1.0

    print("All compute_overall_winrate_by_layoff_period tests passed.")


def test_compute_matchup_winrate_table():
    long_df = pd.DataFrame({
        "own_layoff_period": pd.Categorical(
            ["0-90", "0-90", "91-180"],
            categories=LAYOFF_PERIOD_LABELS,
            ordered=True,
        ),
        "opponent_layoff_period": pd.Categorical(
            ["91-180", "91-180", "0-90"],
            categories=LAYOFF_PERIOD_LABELS,
            ordered=True,
        ),
        "won": [1, 0, 1],
    })

    table = compute_matchup_winrate_table(long_df)

    assert table.loc["0-90", "91-180"] == 0.5
    assert table.loc["91-180", "0-90"] == 1.0

    print("All compute_matchup_winrate_table tests passed.")
