import pandas as pd

from src import plotting, transform


def test_build_weightclass_color_map():
    mock_data = pd.Series({wc: 1 for gender in transform.WEIGHT_DIVISIONS.values() for wc in gender})
    color_map = plotting.build_weightclass_color_map(mock_data)
    all_divisions = transform.WEIGHT_DIVISIONS["male"] + transform.WEIGHT_DIVISIONS["female"]
    for wc in all_divisions:
        assert wc in color_map, f"Missing color for: {wc}"


def test_darken_color():
    color = (1.0, 1.0, 1.0, 1.0)
    darkened = plotting.darken_color(color, factor=0.6)
    assert darkened == (0.6, 0.6, 0.6, 1.0)
    assert darkened[3] == color[3], "Alpha channel should not be darkened."

