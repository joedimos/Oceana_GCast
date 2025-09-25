using Oceananigans
using Oceananigans.Units: hour
using JSON3
using Dates

# Load the GraphCast interface
include("../src/graphcast_interface.jl")
using .GraphCastInterface

# Load utility functions
include("../src/utils.jl")
using .Utils

# Load data processing functions
include("../src/data_processing.jl")
using .DataProcessing

# Define simulation parameters
const SIMULATION_CONFIG = Dict(
    :duration_hours => 48,
    :time_step_hours => 6,
    :output_interval_hours => 1,
    :initial_conditions_file => "../data/era5_sample.nc",
    :synthetic_data_dir => "../data/synthetic_data"
)

function run_simulation()
    # Initialize the GraphCast interface
    iface = create_graphcast_interface(
        (lon_min=-84.3, lon_max=-83.7),
        (lat_min=40.1, lat_max=40.9)
    )

    # Initialize the GraphCast model
    if !initialize_graphcast_model!(iface)
        println("Failed to initialize GraphCast model.")
        return
    end

    # Load initial conditions
    initial_data = fetch_era5_input_data(iface, now())
    if initial_data === nothing
        println("Failed to load initial conditions.")
        return
    end

    # Run the simulation
    for hour in 0:SIMULATION_CONFIG[:duration_hours] / SIMULATION_CONFIG[:time_step_hours] - 1
        lead_time = hour * SIMULATION_CONFIG[:time_step_hours]
        forecast_data = run_graphcast_prediction!(iface, initial_data, lead_time)

        # Save or process forecast data
        if forecast_data !== nothing
            save_forecast_data(forecast_data, lead_time)
        end

        # Update the initial data for the next iteration
        initial_data = forecast_data
    end
end

function save_forecast_data(forecast_data, lead_time)
    # Save the forecast data to a file or database
    filename = "forecast_lead_time_$(lead_time).json"
    JSON3.write(filename, forecast_data)
    println("Saved forecast data for lead time $lead_time hours to $filename")
end

# Execute the simulation
run_simulation()
