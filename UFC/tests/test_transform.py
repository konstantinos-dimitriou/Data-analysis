import numpy as np
import pandas as pd
import pytest

from src import transform


NORMALIZATION_CASES = {
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

    # Women's divisions
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


@pytest.mark.parametrize("raw, expected", NORMALIZATION_CASES.items())
def test_weight_class_normalization(raw, expected):
    result = transform.normalize_weight_classes(raw, weight_divisions=transform.WEIGHT_DIVISIONS)
    assert result == expected


def test_clean_weight_classes():
    # Series path
    raw_series = pd.Series({
        "UFC Interim Lightweight Title":      10,
        "Lightweight":                         5,
        "TUF Lightweight":                     3,
        "UFC 13 Heavyweight Tournament Title": 2,
        "Open Weight":                         1,
    })
    cleaned_series = transform.clean_weight_classes(raw_series)
    assert "TUF Lightweight" not in cleaned_series.index, "TUF entry should be removed."
    assert "Open Weight" not in cleaned_series.index, "Open Weight should be removed."
    assert "UFC 13 Heavyweight Tournament Title" not in cleaned_series.index, "Tournament title should be removed."
    assert cleaned_series["Lightweight"] == 15, "Lightweight entries should be merged and summed."

    # DataFrame path
    raw_df = pd.DataFrame({
        "weight_class": ["UFC Interim Lightweight Title", "Lightweight", "TUF Lightweight", "Open Weight"],
        "method":       ["KO/TKO",                        "KO/TKO",      "KO/TKO",          "KO/TKO"],
        "count":        [10,                               5,             3,                 1],
    })
    cleaned_df = transform.clean_weight_classes(raw_df)
    assert "TUF Lightweight" not in cleaned_df["weight_class"].values, "TUF entry should be removed."
    assert "Open Weight" not in cleaned_df["weight_class"].values, "Open Weight should be removed."
    assert cleaned_df[cleaned_df["weight_class"] == "Lightweight"]["count"].values[0] == 15, \
        "Lightweight entries should be merged and summed."


def test_extract_event_dates():
    medium_df = pd.DataFrame({
        "event": ["Event A", "Event A", "Event B", "Event B", "Event C"],
        "date":  ["1/1/2020", "1/1/2020", "2/2/2020", "2/2/2020", "3/3/2020"],
    })
    result = transform.extract_event_dates(medium_df)
    expected = {"Event A": "1/1/2020", "Event B": "2/2/2020", "Event C": "3/3/2020"}
    assert result == expected


def test_extract_event_dates_inconsistent_dates_raises():
    bad_df = pd.DataFrame({
        "event": ["Event A", "Event A"],
        "date":  ["1/1/2020", "1/2/2020"],
    })
    with pytest.raises(ValueError):
        transform.extract_event_dates(bad_df)


def test_merge_large_and_medium_data():
    # medium dataset: source of dates + draws
    medium_df = pd.DataFrame({
        "event": ["UFC A", "UFC A", "UFC B"],
        "r_fighter": ["F1", "F2", "F3"],
        "b_fighter": ["F4", "F5", "F6"],
        "date": ["01/01/2020", "01/01/2020", "02/01/2020"],
        "status": ["win", "draw", "draw"],
    })
    # large dataset: decisive fights only
    large_df = pd.DataFrame({
        "event_name": ["UFC A", "UFC B"],
        "r_fighter": ["F1", "F3"],
        "b_fighter": ["F4", "F6"],
        "winner": ["Red", "Blue"],
        "fight_order": [0, 0]
    })

    result = transform.merge_large_and_medium_data(large_df, medium_df)

    assert set(result.columns) == {"event_name", "r_fighter", "b_fighter", "winner", "date", "fight_order"}
    assert len(result) == 4, "2 decisive fights + 2 draws."
    assert (result["winner"] == "Draw").sum() == 2
    assert (result["winner"] == "Red").sum() == 1
    assert (result["winner"] == "Blue").sum() == 1
    assert result["date"].notna().all()


def test_add_rest_days_columns():
    df = pd.DataFrame({
        "r_fighter": ["Alice", "Bob",   "Alice"],
        "b_fighter": ["Carol", "Alice", "Dave"],
        "winner":    ["Red",   "Blue",  "Red"],
        "date":      ["3/1/2024", "2/1/2024", "1/1/2024"],
        "fight_order": [0, 0, 0]
    })

    result = transform.add_rest_days_columns(df)

    # Alice, 3/1 (red): previous fight 2/1 = 29 days
    assert result.loc[0, "r_rest_days"] == 29
    # Carol's only fight
    assert pd.isna(result.loc[0, "b_rest_days"])
    # Bob's only fight
    assert pd.isna(result.loc[1, "r_rest_days"])
    # Alice, 2/1 (blue): previous fight 1/1 = 31 days
    assert result.loc[1, "b_rest_days"] == 31
    # Alice's earliest fight
    assert pd.isna(result.loc[2, "r_rest_days"])
    # Dave's only fight
    assert pd.isna(result.loc[2, "b_rest_days"])

"""
def test_exclude_same_night_repeat_fights():
    df = pd.DataFrame({
        "date": ["01/01/2020", "01/01/2020"],
        "r_fighter": ["A", "A"],
        "b_fighter": ["B", "C"],
        "r_rest_days": [100, 0],
        "b_rest_days": [50, 60],
    })

    result = transform.exclude_same_night_repeat_fights(df)

    assert result.loc[0, "r_rest_days"] == 100
    assert np.isnan(result.loc[1, "r_rest_days"]), "Second fight on same night should be NaN."
    assert result.loc[0, "b_rest_days"] == 50
    assert result.loc[1, "b_rest_days"] == 60
"""
def test_exclude_same_night_repeat_fights():
    # Main event first: row 0 is the final (fight_order 1), row 1 the opener (fight_order 0).
    df = pd.DataFrame({
        "date": ["01/01/2020", "01/01/2020"],
        "r_fighter": ["A", "A"],
        "b_fighter": ["B", "C"],
        "r_rest_days": [0, 100],
        "b_rest_days": [50, 60],
        "fight_order": [1, 0],
    })

    result = transform.exclude_same_night_repeat_fights(df)

    assert np.isnan(result.loc[0, "r_rest_days"]), "The final is a same-night repeat."
    assert result.loc[1, "r_rest_days"] == 100, "The opener keeps the layoff."
    assert result.loc[0, "b_rest_days"] == 50
    assert result.loc[1, "b_rest_days"] == 60


def test_same_night_layoff_lands_on_first_fight_across_corners():
    # A fought on night 1, then twice on night 2: blue in the opener, red in the final.
    # Rows are listed main event first, newest night first.
    df = pd.DataFrame({
        "event_name": ["Night 2", "Night 2", "Night 1"],
        "r_fighter": ["A", "C", "A"],
        "b_fighter": ["B", "A", "X"],
        "winner": ["Red", "Blue", "Red"],
        "date": ["02/01/2020", "02/01/2020", "01/01/2020"],
        "fight_order": [1, 0, 0],
    })

    result = transform.add_rest_days_columns(df)
    result = transform.exclude_same_night_repeat_fights(result)

    assert result.loc[1, "b_rest_days"] == 31, "Layoff belongs to A's opener (blue corner)."
    assert np.isnan(result.loc[0, "r_rest_days"]), "A's final is a same-night repeat."



def test_categorize_rest_days():
    rest_days = pd.Series([30, 120, 250, 330, 500, 700, np.nan])
    result_values = transform.categorize_rest_days(rest_days)
    expected_values = ["0-90", "91-180", "181-270", "271-360", "361-540", "540+", np.nan]

    for result, expected in zip(result_values.astype(object), expected_values):
        if pd.isna(expected):
            assert pd.isna(result), "NaN should remain NaN."
        else:
            assert result == expected


def test_compute_layoff_period_matchups():
    df = pd.DataFrame({
        "winner": ["Red", "Blue", "draw"],
        "r_rest_days": [30, 120, 30],
        "b_rest_days": [120, 30, 30],
    })

    result = transform.compute_layoff_period_matchups(df)

    assert len(result) == 4, "Draw should be removed, leaving two fights -> four fighter rows."
    assert result["won"].sum() == 2, "Exactly two fighters should be winners."
    assert set(result["own_layoff_period"].astype(str)) == {"0-90", "91-180"}

