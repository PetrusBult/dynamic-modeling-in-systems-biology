"""
    ObservedData

Structure to hold observed data for a single variable.

# Fields
- `times::Vector{Float64}`: Timepoints where observations were made
- `values::Vector{Float64}`: Observed values at those timepoints
- `weights::Vector{Float64}`: Weights for each observation (e.g., 1/std²)
- `state_index::Int`: Index of the state variable in the ODE solution

# Constructor
```julia
ObservedData(times, values, state_index; weights=ones(length(times)))
```
"""
struct ObservedData
    times::Vector{Float64}
    values::Vector{Float64}
    weights::Vector{Float64}
    state_index::Int
    
    # Constructor with validation
    function ObservedData(times::Vector{Float64}, 
                         values::Vector{Float64}, 
                         state_index::Int;
                         weights::Vector{Float64}=ones(length(times)))
        length(times) == length(values) || 
            throw(ArgumentError("times and values must have same length"))
        length(times) == length(weights) || 
            throw(ArgumentError("times and weights must have same length"))
        state_index > 0 || 
            throw(ArgumentError("state_index must be positive"))
        
        new(times, values, weights, state_index)
    end
end

"""
Internal function to parse CLL data from CSV file.

The CSV format expected:
- Compartment name (e.g., "tissue" or "blood")
- Header line "time, cell count"
- Data rows with time and cell count values
- Empty line between compartments

Returns: Dict{String, NamedTuple} with compartment names as keys
"""
function _parse_cll_csv(filepath::String)
    lines = readlines(filepath)
    
    compartments = Dict{String, NamedTuple}()
    current_compartment = nothing
    times = Float64[]
    values = Float64[]
    
    for line in lines
        # Skip empty lines or header lines
        stripped = strip(line)
        if isempty(stripped) || occursin("time, cell count", stripped)
            continue
        end
        
        # Check if this is a compartment name (no comma)
        if !occursin(',', stripped)
            # Save previous compartment if it exists
            if !isnothing(current_compartment) && !isempty(times)
                compartments[current_compartment] = (times=copy(times), values=copy(values))
            end
            
            # Start new compartment
            current_compartment = stripped
            empty!(times)
            empty!(values)
        else
            # Parse data line
            parts = split(stripped, ',')
            if length(parts) == 2
                push!(times, parse(Float64, strip(parts[1])))
                push!(values, parse(Float64, strip(parts[2])))
            end
        end
    end
    
    # Don't forget the last compartment
    if !isnothing(current_compartment) && !isempty(times)
        compartments[current_compartment] = (times=copy(times), values=copy(values))
    end
    
    return compartments
end

"""
    load_cll_data(filepath::String; 
                  tissue_index::Int=1, 
                  blood_index::Int=2,
                  weights=nothing)

Load CLL data from CSV file and convert to ObservedData structures.

# Arguments
- `filepath::String`: Path to the CSV file
- `tissue_index::Int`: State index for tissue compartment (default: 1)
- `blood_index::Int`: State index for blood compartment (default: 2)
- `weights`: Optional weights. Can be:
  - `nothing`: Use uniform weights (default)
  - `Dict{String, Vector{Float64}}`: Custom weights per compartment
  - `Function`: Function that takes values and returns weights (e.g., `v -> 1 ./ v.^2` for relative weighting)

# Returns
- `Dict{String, ObservedData}`: Dictionary with "tissue" and "blood" keys

# Example
```julia
# Load with uniform weights
observed = load_cll_data("dataset1.csv")

# Load with custom weights
weights = Dict("tissue" => [1.0, 1.0, 1.0], "blood" => ones(17))
observed = load_cll_data("dataset1.csv", weights=weights)

# Load with relative weighting (inversely proportional to squared values)
observed = load_cll_data("dataset1.csv", weights = v -> 1 ./ (v .^ 2))
```
"""
function load_cll_data(filepath::String; 
                       tissue_index::Int=1, 
                       blood_index::Int=2,
                       weights=nothing)
    raw_data = _parse_cll_csv(filepath)
    observed = Dict{String, ObservedData}()
    
    for (compartment, data) in raw_data
        # Determine state index
        state_idx = compartment == "tissue" ? tissue_index : blood_index
        
        # Determine weights
        if isnothing(weights)
            w = ones(length(data.times))
        elseif isa(weights, Dict)
            w = haskey(weights, compartment) ? weights[compartment] : ones(length(data.times))
        elseif isa(weights, Function)
            w = weights(data.values)
        else
            w = ones(length(data.times))
        end
        
        observed[compartment] = ObservedData(data.times, data.values, state_idx, weights=w)
    end
    
    return observed
end
