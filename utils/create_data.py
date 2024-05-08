import numpy as np
import pandas as pd

def generate_facility_location_instance(n_facilities, n_customers, grid_size):
    # Generate random floating-point coordinates for facilities and customers
    facilities = pd.DataFrame({
        "Grid_Lat": np.random.uniform(0, grid_size, n_facilities),
        "Grid_Lon": np.random.uniform(0, grid_size, n_facilities),
        "is_built": 0
    })

    customers = pd.DataFrame({
        "Pop_Lat": np.random.uniform(0, grid_size, n_customers),
        "Pop_Lon": np.random.uniform(0, grid_size, n_customers),
        "population": np.random.poisson(15, n_customers) * np.random.poisson(10, n_customers)
    })

    # Create arrays of facility and customer coordinates for vectorized operations
    facility_coords = facilities[["Grid_Lon", "Grid_Lat"]].to_numpy()
    customer_coords = customers[["Pop_Lon", "Pop_Lat"]].to_numpy()

    # Create the service DataFrame to hold squared distances
    service = pd.DataFrame(index=customers.index, columns=facilities.index, dtype=float)

    # Compute squared distances in a vectorized manner
    for fac_idx, facility in enumerate(facility_coords):
        diff = customer_coords - facility  # Array of differences
        squared_distances = (diff ** 2).sum(axis=1)  # Squared Euclidean distances
        service.iloc[:, fac_idx] = squared_distances

    return service, customers, facilities
