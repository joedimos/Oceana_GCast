module OceananigansUtils

# Utility function to calculate the mean of an array
function mean(arr::AbstractArray{T}) where T
    return sum(arr) / length(arr)
end

# Utility function to calculate the standard deviation of an array
function stddev(arr::AbstractArray{T}) where T
    m = mean(arr)
    return sqrt(sum((arr .- m).^2) / (length(arr) - 1))
end

# Utility function to normalize an array
function normalize(arr::AbstractArray{T}) where T
    μ = mean(arr)
    σ = stddev(arr)
    return (arr .- μ) ./ σ
end

# Utility function to load a NetCDF file and return its contents
function load_netcdf(file_path::String)
    using NetCDF
    nc = NetCDF.File(file_path, "r")
    data = Dict{String, Any}()
    
    for var in names(nc)
        data[var] = nc[var][:]
    end
    
    close(nc)
    return data
end

# Utility function to save data to a NetCDF file
function save_netcdf(file_path::String, data::Dict{String, AbstractArray})
    using NetCDF
    nc = NetCDF.File(file_path, "c")
    
    for (var, values) in data
        nc[var] = values
    end
    
    close(nc)
end

end # module OceananigansUtils
