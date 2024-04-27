import pandas as pd
import numpy as np
from single_step import single_step_area
from visuializer import show_facility_location

data_basic = pd.read_excel("data/Data Ermera Timor Leste.xlsx", sheet_name=None)

#data constants
distances = data_basic["Distances"]
homes = data_basic["Homes"]
locations = data_basic["Potential locations"]
periods_df = pd.DataFrame([1,2,3,4,5,6])
periods = periods_df[0]
per_period = 1
max_distance = 3


Z, X, building = single_step_area(distances, homes, locations, periods, per_period, max_distance)

xC = homes['lon']
yC = homes['lat']
xF = locations['lon']
yF = locations['lat']

show_facility_location(xC, yC, xF, yF, X=X, Z=Z, homes=homes, locations=locations, periods=periods)