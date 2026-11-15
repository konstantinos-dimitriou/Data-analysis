import numpy as np
import pandas as pd


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





