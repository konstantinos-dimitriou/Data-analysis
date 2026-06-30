import pandas as pd

import datetime


date_1 = "11/4/2023"
date_2 = "9/14/2023"
date_1_array = date_1.split("/")
date_2_array = date_2.split("/")

date_1 = datetime.datetime(int(date_1_array[2]), int(date_1_array[0]), int(date_1_array[1]))
date_2 = datetime.datetime(int(date_2_array[2]), int(date_2_array[0]), int(date_2_array[1]))
diff_in_days = date_2 - date_1
#print(diff_in_days.days)

indexes = [0,1,2,3,4]
for i in indexes:
    print("In i loop ",i)
    for j in indexes:
        if i >= j:
            continue
        print("i ",i, " j ",j)





"""
a_dic = {"Teams": ["OSFP", "PAO"], "Stadium": ["OAKA", "OLYMPIC"], "Cars": ["OPEL", "VW"]}
b_dic = {"Teams": ["OSFP", "PAO"], "Stadium": ["OLYMPIC", "OAKA"], "Animal": ["LION", "TIGER"]}
a_df = pd.DataFrame(a_dic)
b_df = pd.DataFrame(b_dic)
print(a_df)
print(b_df)

a_b_merged = a_df.merge(b_df, on="Teams")
print(a_b_merged)
a_b_merged = a_b_merged.drop("Stadium_y", axis=1)
print(a_b_merged)
a_b_merged = a_b_merged.rename(columns={"Stadium_x": "Stadium"})
print(a_b_merged)
"""

pd.set_option("display.max_rows", None)

large_dataframe = pd.read_csv("DATA/Large_set/large_dataset.csv")
medium_dataframe = pd.read_csv("DATA/Medium_set/medium_dataset.csv")

L_DF = large_dataframe[["event_name","r_fighter","b_fighter","winner"]]
#M_DF = medium_dataframe[["event", "r_fighter","b_fighter", "date"]]
M_DF = medium_dataframe[["event", "date"]]
M_DF = M_DF.rename(columns={"event": "event_name"})
#print(L_DF[:10])
#print(M_DF[:10])
merged_df = L_DF.merge(M_DF, on="event_name")

merged_df = merged_df.drop_duplicates()
merged_df = merged_df.reset_index(drop=True)
#print(merged_df[:10])
#merged_df.info()

#concatinated_L_M_DF = pd.concat([L_DF,M_DF], axis=1)
#concatinated_L_M_DF = pd.concat([L_DF,M_DF], axis=0)
#print(concatinated_L_M_DF[:10])
#print(concatinated_L_M_DF[:1])

#joined_L_M_DF = L_DF.join(M_DF, on="event_name")
#print(joined_L_M_DF)

def custom_combine_dataframes(df_1, df_2):

    pass
