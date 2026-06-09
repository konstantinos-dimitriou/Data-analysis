import pandas as pd
from ufc_events_analysis import (
    normalize_weight_classes,
    clean_weight_classes,
    derive_ko_totals,
    build_weightclass_color_map,
    darken_color,
    WEIGHT_DIVISIONS
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