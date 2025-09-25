using Oceananigans
using NetCDF
using DataFrames
using JSON3

# Function to load NetCDF data
function load_netcdf_data(file_path::String)
    if !isfile(file_path)
        error("File not found: $file_path")
    end
    
    dataset = NetCDF.Dataset(file_path)
    data = Dict{String, Any}()
    
    for var in keys(dataset.variables)
        data[var] = dataset.variables[var][:]
    end
    
    return data
end

# Function to preprocess synthetic data
function preprocess_synthetic_data(data::Dict{String, Any})
    # Example transformation: normalize temperature data
    if haskey(data, "temperature")
        data["temperature"] .= (data["temperature"] .- mean(data["temperature"])) ./ std(data["temperature"])
    end
    
    return data
end

# Main function to preprocess data
function preprocess_data()
    era5_data = load_netcdf_data("../data/era5_sample.nc")
    synthetic_data = load_netcdf_data("../data/synthetic_data/synthetic_data.nc")  # Adjust path as necessary
    
    processed_era5_data = preprocess_synthetic_data(era5_data)
    processed_synthetic_data = preprocess_synthetic_data(synthetic_data)
    
    return processed_era5_data, processed_synthetic_data
end

# Execute preprocessing
if abspath(PROGRAM_FILE) == abspath(__FILE__)
    processed_era5, processed_synthetic = preprocess_data()
    println("Preprocessing completed.")
end
