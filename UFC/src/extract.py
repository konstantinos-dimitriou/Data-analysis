import pandas as pd


def read_large_dataset(path):
    return pd.read_csv(path)


def read_medium_dataset(path):
    return pd.read_csv(path)


def check_data_completeness(df):
    data_entry_numbers = df.isnull().sum()
    data_holes = data_entry_numbers[data_entry_numbers !=0]
    if len(data_holes) != 0:
        print("Holes in data:")
        print(data_holes)
        return False
    else:
        return True

    