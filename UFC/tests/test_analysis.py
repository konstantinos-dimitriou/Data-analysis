import pandas as pd

from src import analysis, transform


def test_derive_ko_totals():
    raw_df = pd.DataFrame({
        "weight_class": ["Lightweight", "Lightweight", "Lightweight", "Heavyweight", "Heavyweight"],
        "method":       ["KO/TKO",      "Submission",  "KO/TKO",      "KO/TKO",      "Decision - Unanimous"],
        "count":        [10,             5,             3,             8,             4],
    })

    total_fights, ko_fights = analysis.derive_ko_totals(raw_df)

    assert total_fights["Lightweight"] == 18
    assert total_fights["Heavyweight"] == 12
    assert ko_fights["Lightweight"] == 13
    assert ko_fights["Heavyweight"] == 8
    assert list(total_fights.index) == ["Lightweight", "Heavyweight"], "Sorted descending by fight count."
    assert list(ko_fights.index) == list(total_fights.index), "ko_fights follows the order of total_fights."


def test_derive_ko_totals_no_ko_returns_zero():
    raw_df = pd.DataFrame({
        "weight_class": ["Flyweight"],
        "method":       ["Submission"],
        "count":        [5],
    })
    _, ko_fights = analysis.derive_ko_totals(raw_df)
    assert ko_fights["Flyweight"] == 0, "No KO/TKO should return 0, not NaN."


def test_derive_corner_win_counts():
    merged_df = pd.DataFrame({
        "date":   ["01/01/2020", "01/01/2020", "02/01/2020", "03/01/2020"],
        "winner": ["Red",        "Blue",       "Draw",       "Red"],
    })

    # no cutoff
    red, blue, draws = analysis.derive_corner_win_counts(merged_df)
    assert (red, blue, draws) == (2, 1, 1)

    # cutoff excludes the first two fights
    red, blue, draws = analysis.derive_corner_win_counts(merged_df, cutoff_date="01/15/2020")
    assert (red, blue, draws) == (1, 0, 1)


def test_cutoff_excludes_all():
    df = pd.DataFrame({"date": ["01/01/2020"], "winner": ["Red"]})
    red, blue, draws = analysis.derive_corner_win_counts(df, cutoff_date="02/01/2020")
    assert (red, blue, draws) == (0, 0, 0)


def test_compute_weeks_off_distribution():
    df = pd.DataFrame({
        "winner":      ["Red", "Blue", "Red"],
        "r_rest_days": [14,    None,   21],
        "b_rest_days": [None,  35,     None],
    })

    # All fighters, NaN dropped
    result = analysis.compute_weeks_off_distribution(df, winners_only=False)
    assert result.get(2, 0) == 1
    assert result.get(3, 0) == 1
    assert result.get(5, 0) == 1
    assert result.sum() == 3, "NaN values should not be counted, not even as 0 weeks off."

    # Winners only: Red won fights 0 and 2, Blue won fight 1
    result_winners = analysis.compute_weeks_off_distribution(df, winners_only=True)
    assert result_winners.get(2, 0) == 1
    assert result_winners.get(3, 0) == 1
    assert result_winners.get(5, 0) == 1
    assert result_winners.sum() == 3


def test_compute_overall_winrate_by_layoff_period():
    long_df = pd.DataFrame({
        "own_layoff_period": pd.Categorical(
            ["0-90", "0-90", "91-180", "91-180"],
            categories=transform.LAYOFF_PERIOD_LABELS,
            ordered=True,
        ),
        "won": [1, 0, 1, 1],
    })

    result = analysis.compute_overall_winrate_by_layoff_period(long_df)

    assert result.loc["0-90"] == 0.5
    assert result.loc["91-180"] == 1.0


def test_compute_matchup_winrate_table():
    long_df = pd.DataFrame({
        "own_layoff_period": pd.Categorical(
            ["0-90", "0-90", "91-180"],
            categories=transform.LAYOFF_PERIOD_LABELS,
            ordered=True,
        ),
        "opponent_layoff_period": pd.Categorical(
            ["91-180", "91-180", "0-90"],
            categories=transform.LAYOFF_PERIOD_LABELS,
            ordered=True,
        ),
        "won": [1, 0, 1],
    })

    table = analysis.compute_matchup_winrate_table(long_df)

    assert table.loc["0-90", "91-180"] == 0.5
    assert table.loc["91-180", "0-90"] == 1.0

