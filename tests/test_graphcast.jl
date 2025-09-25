using Test
using Oceananigans
using GraphCastInterface  # Assuming this is the module name for graphcast_interface.jl

# Test for GraphCast Interface
function test_load_model()
    iface = GraphCastInterface()
    success = initialize_graphcast_model!(iface)
    @test success == true
end

function test_fetch_era5_data()
    iface = GraphCastInterface()
    initialize_graphcast_model!(iface)
    target_time = DateTime(2024, 1, 1, 0, 0)
    data = fetch_era5_input_data(iface, target_time)
    @test data !== nothing
end

function test_run_prediction()
    iface = GraphCastInterface()
    initialize_graphcast_model!(iface)
    target_time = DateTime(2024, 1, 1, 0, 0)
    input_data = fetch_era5_input_data(iface, target_time)
    lead_time_hours = 6
    forecast = run_graphcast_prediction!(iface, input_data, lead_time_hours)
    @test forecast !== nothing
end

# Run all tests
@testset "GraphCast Interface Tests" begin
    test_load_model()
    test_fetch_era5_data()
    test_run_prediction()
end
