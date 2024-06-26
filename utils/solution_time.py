import timeit
import numpy as np
import pandas as pd
from create_data import generate_facility_location_instance

# Create function used for meassuring computation times
def measure_time(func, distances, homes, locations, periods, per_period, max_distance, open_facilities, vpop, num_passes=10):

    timer = timeit.Timer(lambda: func(distances, homes, locations, periods, per_period, max_distance, open_facilities, vpop))
    times = timer.repeat(repeat=num_passes, number=1)
    min_time = np.min(times)
    avg_time = np.mean(times)
    return min_time , avg_time

def get_solution_times(functions, params, num_passes=10):

    # Prepare to collect results
    results = []

    # Set parameters
    periods_df = pd.DataFrame([1,2,3,4,5,6,7,8,9,10])
    periods = periods_df[0]
    per_period = 2
    max_distance = 3

    # Repeat for tuples of households:locations
    for (no_of_households, no_of_facilities) in params:
        # Generate the data
        service, customers, facilities  = generate_facility_location_instance(no_of_facilities, no_of_households, 20)

        # Set remaining parameters
        distances = service.to_numpy()
        homes = customers
        locations = facilities
        open_facilities = facilities["is_built"]
        vpop = customers["population"]

        # Solve 10 times with each function and collect results
        for func in functions:
            min_time, avg_time = measure_time(func, distances, homes, locations, periods, per_period, max_distance, open_facilities, vpop, num_passes)
            results.append({
                "Function": func.__name__,
                "Households": no_of_households,
                "Facilities": no_of_facilities,
                "Min Time(s)": min_time,
                "Avg Time(s)": avg_time
            })

    return pd.DataFrame(results)