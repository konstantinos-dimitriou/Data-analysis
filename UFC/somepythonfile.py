import pandas as pd

df = pd.read_csv("data/processed/fights_with_rest_days.csv")
night = df[df["date"] == "03/11/1994"]
print(night[["event_name", "r_fighter", "b_fighter", "winner", "r_rest_days", "b_rest_days"]])