module SimODEs_BioSB

# Include the simulation functionality
include("ode_models.jl")
include("simulate.jl")
include("data_loading.jl")
include("plotting.jl")
include("estimation.jl")

# Export the abstract type
export AbstractODEModel, WodartzModel

# Export the main simulation function
export simulate

# Export data structures
export ObservedData

# Export data loading functions
export load_cll_data

# Export plotting functions
export plot_cll

# Export estimation functions
export weighted_sse, create_loss_function, MultiStartConfig, fit_model, merge_parameters, fit_model_crs, CRSConfig

end # module SimODEs_BioSB