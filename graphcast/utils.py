def load_data(file_path):
    import numpy as np
    import xarray as xr

    try:
        # Load NetCDF data using xarray
        data = xr.open_dataset(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def preprocess_data(data):
    # Example preprocessing function
    if data is None:
        return None

    # Perform some preprocessing steps
    processed_data = data.sel(time=slice("2020-01-01", "2020-12-31"))  # Example: filter by time
    return processed_data

def save_processed_data(data, output_path):
    try:
        # Save processed data to a new NetCDF file
        data.to_netcdf(output_path)
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")

def generate_synthetic_data(shape):
    # Generate synthetic data for testing
    return np.random.rand(*shape)  # Example: random data with given shape

def normalize_data(data):
    # Normalize data to the range [0, 1]
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val) if max_val > min_val else data
