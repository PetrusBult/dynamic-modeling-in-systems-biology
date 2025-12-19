"""
    simulate(model::AbstractODEModel; tspan, alg=Tsit5(), kwargs...)

Simulate an ODE model over a time span.

# How it works
1. model.parameters -> params for ODEProblem
2. model.initial_conditions -> initial state vector
3. ode_func!(model, ...) -> the ODE model equations
4. solve() -> solution

# Arguments
- `model::AbstractODEModel`: The ODE model to simulate
- `tspan`: Time span as a tuple (t_start, t_end), e.g., (0.0, 100.0)
- `alg`: ODE solver algorithm (default: Tsit5())
- `kwargs...`: Additional keyword arguments passed to the solver

# Returns
- ODE solution object

# Example
sol = simulate(LotkaVolterraModel(), tspan=(0.0, 50.0))
sol = simulate(MixedMealModel(), tspan=(0.0, 480.0))  # 8 hours in minutes

# For custom parameters, create a new model instance:
custom_params = (alpha=1.5, beta=0.5, delta=0.5, gamma=0.2)
custom_ic = (prey=15.0, predator=3.0)
custom_model = LotkaVolterraModel(custom_params, custom_ic)
sol = simulate(custom_model, tspan=(0.0, 100.0))
"""
function simulate(model::AbstractODEModel; tspan, alg=Tsit5(), kwargs...)
    # Validate inputs
    length(tspan) == 2 || throw(ArgumentError("tspan must be a 2-element tuple (t_start, t_end)"))
    tspan[1] < tspan[2] || throw(ArgumentError("tspan must be increasing: tspan[1] < tspan[2]"))
    
    # Extract parameters and initial conditions
    p = model.parameters
    u0 = MVector(values(model.initial_conditions))

    # Define the ODE problem
    prob = ODEProblem((du, u, p, t) -> ode_func!(model, du, u, p, t), u0, tspan, p; kwargs...)

    # Solve the ODE problem
    sol = solve(prob, alg; kwargs...)

    return sol
end

"""
    simulate(models::Vector{<:AbstractODEModel}; tspan, kwargs...)

Simulate multiple ODE models with the same parameters.
"""
function simulate(models::Vector{<:AbstractODEModel}; kwargs...)
    return [simulate(model; kwargs...) for model in models]
end
