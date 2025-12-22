using Statistics
using Optimization, OptimizationOptimJL
using Random
using Sobol
using QuasiMonteCarlo

# ============================================================================
# Functions regarding loss functions
# ============================================================================
"""
    weighted_sse(model::AbstractODEModel, 
                 params_to_estimate::NamedTuple,
                 observed_data::Vector{ObservedData},
                 tspan::Union{Tuple{Float64, Float64}, Nothing}=nothing) -> Float64

    Compute weighted sum of squared errors between model predictions and observations.

    # Arguments
    - `model::AbstractODEModel`: The ODE model
    - `params_to_estimate::NamedTuple`: Subset of parameters being estimated
    - `observed_data::Vector{ObservedData}`: Observed data for each variable
    - `tspan::Union{Tuple{Float64, Float64}, Nothing}`: Time span for simulation
      - If `nothing`, automatically inferred as `(0.0, maximum(observation_times))`
      - If provided, must be a 2-tuple `(t_start, t_end)`

    # Returns
    - `Float64`: Weighted sum of squared errors
"""
function weighted_sse(model::AbstractODEModel, 
                     params_to_estimate::NamedTuple,
                     observed_data::Vector{ObservedData},
                     tspan::Union{Tuple{Float64, Float64}, Nothing}=nothing)
    # Validate input
    isempty(observed_data) && throw(ArgumentError("observed_data cannot be empty"))
    
    # Merge estimated parameters with model's base parameters
    full_params = merge_parameters(model.parameters, params_to_estimate)
    
    # Create model with updated parameters
    updated_model = typeof(model)(full_params, model.initial_conditions)
    
    # Collect all unique timepoints and state indices
    all_times = sort(unique(vcat([obs.times for obs in observed_data]...)))
    all_state_indices = sort(unique([obs.state_index for obs in observed_data]))
    
    # Validate state indices exist in model
    max_state_idx = maximum(all_state_indices)
    max_state_idx <= length(model.initial_conditions) || 
        throw(ArgumentError("state_index $max_state_idx exceeds model states ($(length(model.initial_conditions)))"))
    
    # Infer tspan if not provided: always start at 0.0, end at last observation
    if tspan === nothing
        t_max = maximum(maximum(obs.times) for obs in observed_data)
        tspan = (0.0, t_max)
    end
    
    # Simulate only at observation timepoints and only save observed states
    sol = simulate(updated_model; tspan=tspan, saveat=all_times, save_idxs=all_state_indices)
    
    # Check if simulation was successful (use SciMLBase helper function)
    SciMLBase.successful_retcode(sol) || return Inf  # Return large penalty if solve failed

    # Compute weighted sum of squared errors
    total_sse = 0.0
    for obs_data in observed_data
        # Find which index in save_idxs corresponds to this observable
        saved_idx = findfirst(==(obs_data.state_index), all_state_indices)
        
        for (t, obs_val, weight) in zip(obs_data.times, obs_data.values, obs_data.weights)
            # Get prediction from saved solution
            # sol(t) returns only the saved states, so use saved_idx not state_index
            pred_val = sol(t)[saved_idx]
            residual = pred_val - obs_val
            total_sse += weight * residual^2
        end
    end
    
    return total_sse
end


"""
    create_loss_function(model::AbstractODEModel,
                        observed_data::Vector{ObservedData},
                        param_names::NTuple{N, Symbol},
                        tspan::Union{Tuple{Float64, Float64}, Nothing}=nothing;
                        use_log_transform::Bool=false) where N

    Create a loss function for parameter estimation compatible with Optimization.jl.

    The returned function has signature `loss(params_vector, p)` where:
    - `params_vector`: Vector of parameter values being optimized
    - `p`: Auxiliary data (unused, required by Optimization.jl interface)

    # Arguments
    - `model::AbstractODEModel`: The base ODE model
    - `observed_data::Vector{ObservedData}`: Vector of observed data for each variable
    - `param_names::NTuple{N, Symbol}`: Names of parameters to estimate (as tuple of Symbols)
    - `tspan::Union{Tuple{Float64, Float64}, Nothing}`: Time span for simulation
    - If `nothing` (default), automatically inferred as `(0.0, maximum(observation_times))`
    - If provided, must be a 2-tuple `(t_start, t_end)`
    - `use_log_transform::Bool`: If true, optimize in log-space (useful for positive parameters)

    # Returns
    - Function with signature `loss(params_vector::Vector, p)` compatible with Optimization.jl
"""
function create_loss_function(model::AbstractODEModel,
                             observed_data::Vector{ObservedData},
                             param_names::NTuple{N, Symbol},
                             tspan::Union{Tuple{Float64, Float64}, Nothing}=nothing;
                             use_log_transform::Bool=false) where N
    
    # Closure captures: model, observed_data, param_names, tspan, use_log_transform
    function loss(params_vector::AbstractVector, p)
        # Convert vector to NamedTuple using captured param_names
        if use_log_transform
            # Transform from log-space
            params_nt = NamedTuple{param_names}(Tuple(exp(pv) for pv in params_vector))
        else
            params_nt = NamedTuple{param_names}(Tuple(params_vector))
        end
        
        # Compute weighted SSE
        return weighted_sse(model, params_nt, observed_data, tspan)
    end
    
    return loss
end


"""
    MultiStartConfig(; n_samples=100, n_starts=10, method=:sobol, parallel=true, seed=nothing)

    Configuration for multi-start optimization. Modern Julia pattern using `@kwdef` with validation.

    # Fields
    - `n_samples::Int`: Number of initial parameter sets to sample (default: 100)
    - `n_starts::Int`: Number of best initial points to optimize from (default: 10)
    - `method::Symbol`: Sampling method - `:sobol`, `:lhs`, or `:uniform` (default: `:sobol`)
    - `parallel::Bool`: Use threading for parallel evaluation (default: `true`)
    - `seed::Union{Int, Nothing}`: Random seed for reproducibility (default: `nothing`)

    # Example
    # Use defaults
    ms = MultiStartConfig()

    # Customize
    ms = MultiStartConfig(n_samples=100, n_starts=10, method=:lhs, seed=42)

    # Use in fit_model
    fitted_model, result = fit_model(model, obs_data, (:k1, :k5),
        lb=[0.0, 0.0], ub=[1.0, 1.0], multistart=ms)
"""
Base.@kwdef struct MultiStartConfig
    n_samples::Int = 100
    n_starts::Int = 10
    method::Symbol = :sobol
    parallel::Bool = true
    seed::Union{Int, Nothing} = nothing
    
    # Validation in inner constructor
    function MultiStartConfig(n_samples, n_starts, method, parallel, seed)
        n_samples > 0 || throw(ArgumentError("n_samples must be positive, got $n_samples"))
        n_starts > 0 || throw(ArgumentError("n_starts must be positive, got $n_starts"))
        n_starts <= n_samples || 
            throw(ArgumentError("n_starts ($n_starts) cannot exceed n_samples ($n_samples)"))
        method in (:sobol, :lhs, :uniform) || 
            throw(ArgumentError("method must be :sobol, :lhs, or :uniform, got :$method"))
        new(n_samples, n_starts, method, parallel, seed)
    end
end

# ============================================================================
# Parameter Estimation with Multi-Start & parallelization Support
# ============================================================================

"""
    fit_model(model, observed_data, param_names; 
              lb=nothing, ub=nothing, initial_guess=nothing,
              optimizer=LBFGS(), tspan=nothing, use_log_transform=false,
              multistart=nothing, return_all=false, optimizer_kwargs...)

    Fit model parameters to observed data using gradient-based optimization.

    # Arguments
    - `model::AbstractODEModel`: The ODE model to fit
    - `observed_data::Vector{ObservedData}`: Observed data for each variable
    - `param_names`: Names of parameters to estimate (Tuple or Vector of Symbols)

    # Keyword Arguments
    - `lb`, `ub`: Lower/upper bounds (vectors matching param_names length)
    - `initial_guess`: Starting values (default: current model values)
    - `optimizer`: Optimization algorithm (default: LBFGS())
    - `tspan`: Time span for simulation (default: auto from data)
    - `use_log_transform::Bool`: Optimize in log-space (default: false)
    - `multistart::Union{MultiStartConfig, Nothing}`: Multi-start config (default: single-start)
    - `return_all::Bool`: Return all results (true) or just best (false)
    - `optimizer_kwargs...`: Additional kwargs passed to solve()

    # Returns
    - Single-start or `return_all=false`: `(fitted_model, result)`
    - Multi-start with `return_all=true`: `(Vector{model}, Vector{result})`

    # Examples
    # Single-start with bounds
    fitted_model, result = fit_model(
        model, obs_data, (:k1, :k5),
        lb=[0.0, 0.0], ub=[1.0, 1.0]
    )

    # Multi-start optimization
    ms = MultiStartConfig(n_samples=100, n_starts=10, method=:lhs, seed=42)
    fitted_model, result = fit_model(
        model, obs_data, (:k1, :k5),
        lb=[0.0, 0.0], ub=[1.0, 1.0],
        multistart=ms
    )

    # Get all results
    models, results = fit_model(model, obs_data, (:k1, :k5),
        lb=[0.0, 0.0], ub=[1.0, 1.0],
        multistart=ms, return_all=true
    )  
"""
function fit_model(model::AbstractODEModel,
                  observed_data::Vector{ObservedData},
                  param_names::Union{NTuple{N, Symbol}, Vector{Symbol}};
                  lb=nothing,
                  ub=nothing,
                  initial_guess=nothing,
                  optimizer=LBFGS(),
                  tspan=nothing,
                  use_log_transform::Bool=false,
                  multistart::Union{MultiStartConfig, Nothing}=nothing,
                  return_all::Bool=false,
                  optimizer_kwargs...) where N
    
    # Normalize and validate
    params = param_names isa Tuple ? param_names : Tuple(param_names)
    _validate_param_names(model, params)
    
    guess = _get_initial_guess(model, params, initial_guess)
    lb, ub = _validate_bounds(lb, ub, guess, length(params))
    
    multistart !== nothing && (lb === nothing || ub === nothing) &&
        throw(ArgumentError("Multi-start optimization requires both lb and ub"))
    
    # Create loss function
    loss_fn = create_loss_function(model, observed_data, params, tspan;
                                   use_log_transform=use_log_transform)
    
    # Dispatch to single or multi-start
    if multistart === nothing
        return _fit_single(model, params, loss_fn, guess, lb, ub, optimizer, optimizer_kwargs)
    else
        return _fit_multi(model, params, loss_fn, lb, ub, multistart, optimizer, 
                         return_all, optimizer_kwargs)
    end
end

# ============================================================================
# Helper functions for fit_model
# ============================================================================
"""
    merge_parameters(base::NamedTuple, updates::NamedTuple) -> NamedTuple

    Merge parameter updates into base parameters, preserving order and non-updated values.
"""
function merge_parameters(base::NamedTuple, updates::NamedTuple)::NamedTuple
    # Validate that all update keys exist in base
    update_keys = keys(updates)
    all(k -> k in keys(base), update_keys) || 
        throw(ArgumentError("All update keys must exist in base parameters. " *
                          "Invalid keys: $(setdiff(update_keys, keys(base)))"))
    
    # Build new NamedTuple with same keys and order as base
    new_vals = Tuple(get(updates, k, base[k]) for k in keys(base))
    return NamedTuple{keys(base)}(new_vals)
end

_validate_param_names(model, params) = all(p -> p in keys(model.parameters), params) ||
    throw(ArgumentError("All parameters must exist in model.parameters"))

function _get_initial_guess(model, params, guess)::Vector{Float64}
    if guess === nothing
        return [Float64(model.parameters[k]) for k in params]
    end
    length(guess) == length(params) ||
        throw(ArgumentError("initial_guess length must match param_names"))
    return Vector{Float64}(guess)
end

function _validate_bounds(lb, ub, guess, n)
    lb !== nothing && length(lb) != n && throw(ArgumentError("lb length must match params"))
    ub !== nothing && length(ub) != n && throw(ArgumentError("ub length must match params"))
    
    if lb !== nothing && ub !== nothing
        all(lb .<= ub) || throw(ArgumentError("lb must be <= ub"))
        all(lb .<= guess .<= ub) || @warn "Initial guess outside bounds"
    end
    
    return lb, ub
end

function _build_model(model, params, param_values)
    updates = NamedTuple{params}(Tuple(param_values))
    updated_params = merge_parameters(model.parameters, updates)
    typeof(model)(updated_params, model.initial_conditions)
end

function _solve_optimization(loss_fn, initial_values, lb, ub, optimizer, kwargs)
    opt_f = OptimizationFunction((u, p) -> loss_fn(u, p), Optimization.AutoForwardDiff())
    prob = OptimizationProblem(opt_f, initial_values, nothing; lb=lb, ub=ub)
    Optimization.solve(prob, optimizer; kwargs...)
end

function _fit_single(model, params, loss_fn, guess, lb, ub, optimizer, kwargs)
    result = _solve_optimization(loss_fn, guess, lb, ub, optimizer, kwargs)
    fitted_model = _build_model(model, params, result.u)
    return fitted_model, result
end

function _fit_multi(model, params, loss_fn, lb, ub, config::MultiStartConfig, 
                   optimizer, return_all, kwargs)
    config.seed !== nothing && Random.seed!(config.seed)
    
    # Generate and evaluate initial samples
    samples = _generate_samples(config.method, config.n_samples, lb, ub, length(params))
    losses = _evaluate_samples(loss_fn, samples, config.parallel)
    
    # Select best starting points
    best_idx = sortperm(losses)[1:config.n_starts]
    @info "Optimizing from $(config.n_starts) best starts (losses: $(@views losses[best_idx]))"

    # Optimize from each start
    results = _optimize_from_starts(loss_fn, @view(samples[:, best_idx]), lb, ub, 
                                    optimizer, config.parallel, kwargs)
    
    # Sort and build models
    sorted_results = sort!(results, by=r -> r.minimum)  # Use sort! to mutate in-place
    @info "Best final loss: $(sorted_results[1].minimum)"
    
    fitted_models = [_build_model(model, params, r.u) for r in sorted_results]
    
    return_all ? (fitted_models, sorted_results) : (fitted_models[1], sorted_results[1])
end

function _evaluate_samples(loss_fn, samples::AbstractMatrix, parallel::Bool)::Vector{Float64}
    n = size(samples, 2)
    losses = zeros(Float64, n)
    
    if parallel
        Threads.@threads for i in 1:n
            @inbounds losses[i] = loss_fn(samples[:, i], nothing)
        end
    else
        @inbounds for i in 1:n
            losses[i] = loss_fn(samples[:, i], nothing)
        end
    end
    
    return losses
end

function _optimize_from_starts(loss_fn, starts::AbstractMatrix, lb, ub, optimizer, parallel::Bool, kwargs)
    n_starts = size(starts, 2)
    results = Vector{Any}(undef, n_starts)
    
    if parallel
        Threads.@threads for i in 1:n_starts
            @inbounds results[i] = _solve_optimization(loss_fn, starts[:, i], lb, ub, optimizer, kwargs)
        end
    else
        @inbounds for i in 1:n_starts
            results[i] = _solve_optimization(loss_fn, starts[:, i], lb, ub, optimizer, kwargs)
        end
    end
    
    return results
end

# Sampling methods for multi-start optimization
function _generate_samples(method::Symbol, n, lb, ub, n_params)
    method == :sobol && return _sobol_samples(n, lb, ub, n_params)
    method == :lhs && return QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())
    method == :uniform && return lb .+ rand(n_params, n) .* (ub .- lb)
    throw(ArgumentError("Unknown sampling method: $method"))
end

function _sobol_samples(n, lb, ub, n_params)
    s = SobolSeq(n_params)
    samples = zeros(n_params, n)
    @inbounds for i in 1:n
        samples[:, i] = lb .+ next!(s) .* (ub .- lb)
    end
    return samples
end
