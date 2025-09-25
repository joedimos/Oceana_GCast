using Oceananigans
using Test

include("../src/main.jl")
include("../src/utils.jl")
include("../src/graphcast_interface.jl")
include("../src/data_processing.jl")

# Define a test suite for the simulation functions
@testset "Simulation Tests" begin

    # Test case for initializing the simulation environment
    @testset "Initialization" begin
        # Assuming there's a function `initialize_simulation` in main.jl
        result = initialize_simulation()
        @test result != nothing "Simulation should initialize successfully"
    end

    # Test case for running a simulation
    @testset "Run Simulation" begin
        # Assuming there's a function `run_simulation` in main.jl
        simulation_result = run_simulation()
        @test simulation_result.success "Simulation should run successfully"
        @test simulation_result.output != nothing "Simulation output should not be empty"
    end

    # Test case for data processing
    @testset "Data Processing" begin
        # Assuming there's a function `process_data` in data_processing.jl
        processed_data = process_data("path/to/data")
        @test processed_data != nothing "Processed data should not be empty"
        @test processed_data["temperature"] != nothing "Temperature data should be present"
    end

  

end
