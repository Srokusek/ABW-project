import pandas as pd
import numpy as np

current_facilities = pd.read_csv("./data/ermera_full/current_facs.csv")
current_facilities["is_built"] = 1
current_facilities = current_facilities.drop(columns=["Hosp_Name"])
current_facilities.columns = ["Pop_Lat", "Pop_Lon", "Population", "Grid_Lat", "Grid_Lon", "Distance_KM", "is_built"]

data_potential = pd.read_csv("./data/ermera_full/grid.csv")
data_potential["is_built"] = 0

# Merge data to include built facilities
data = pd.concat([data_potential, current_facilities], ignore_index=True)

# Collect unique households and potential facilities
populations = data[["Pop_Lat", "Pop_Lon", "Population"]].drop_duplicates().reset_index().drop(columns=["index"])
facilities = data[["Grid_Lat", "Grid_Lon", "is_built"]].drop_duplicates().reset_index().drop(columns=["index"])

# Initialize distances matrix
distances = pd.DataFrame(
    columns=facilities.index.astype(int),
    index=populations.index.astype(int),
    data=np.nan
)

# Populate distances matrix
for i, population in populations.iterrows():
    for j, facility in facilities.iterrows():
        matching_row = data[
            (data["Pop_Lat"] == population["Pop_Lat"]) &
            (data["Pop_Lon"] == population["Pop_Lon"]) &
            (data["Grid_Lat"] == facility["Grid_Lat"]) &
            (data["Grid_Lon"] == facility["Grid_Lon"])
        ]

        if not matching_row.empty:
            distances.at[i,j] = matching_row.iloc[0]["Distance_KM"]

# Replace any potentially missing values by a really high distance value
distances.replace(np.nan, 1000, inplace=True)

# Export and save the data for future use
populations.to_csv("./data/ermera_full/processed/populations.csv")
facilities.to_csv("./data/ermera_full/processed/facilities.csv")
distances.to_csv("./data/ermera_full/processed/distances.csv")