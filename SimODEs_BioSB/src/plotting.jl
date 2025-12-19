using Plots

# Compartment configuration
const COMPARTMENTS = (
    tissue = (colors = [:blue, :green, :purple, :cyan, :darkblue], 
              title = "Tissue Compartment"),
    blood  = (colors = [:red, :orange, :brown, :pink, :darkred], 
              title = "Blood Compartment")
)

const MARKERS = [:circle, :square, :diamond, :utriangle, :dtriangle]
const PLOT_DEFAULTS = (
    xlabel = "Time (days)",
    ylabel = "Absolute Lymphocyte Count",
    markersize = 5,
    legend = :topright,
    size = (800, 600)
)

# Helper functions
_calc_errorbars(obs::ObservedData, show::Bool) = show ? (1 ./ sqrt.(obs.weights)) : nothing
_cycle(vec, idx) = vec[mod1(idx, length(vec))]

"""
Add scatter plot to existing plot or create new one.
"""
function _scatter_data!(p, times, values, label; yerror=nothing, color, marker, alpha=0.7)
    kwargs = (label=label, marker=marker, color=color, alpha=alpha,
              xlabel=PLOT_DEFAULTS.xlabel, ylabel=PLOT_DEFAULTS.ylabel,
              markersize=PLOT_DEFAULTS.markersize, legend=PLOT_DEFAULTS.legend)
    
    if !isnothing(yerror)
        kwargs = merge(kwargs, (yerror=yerror,))
    end
    
    isnothing(p) ? scatter(times, values; kwargs...) : 
                   (scatter!(p, times, values; kwargs...); p)
end

"""
Add line plot to existing plot or create new one.
"""
function _plot_line!(p, times, values, label; color, linewidth=2.5, alpha=1.0)
    kwargs = (label=label, color=color, linewidth=linewidth, alpha=alpha,
              xlabel=PLOT_DEFAULTS.xlabel, ylabel=PLOT_DEFAULTS.ylabel,
              legend=PLOT_DEFAULTS.legend)
    
    isnothing(p) ? plot(times, values; kwargs...) : 
                   (plot!(p, times, values; kwargs...); p)
end

"""
Create plot for a single compartment with observed data.
"""
function _plot_compartment(comp_name::String, datasets, labels, show_errorbars::Bool, alpha::Float64)
    comp_key = Symbol(comp_name)
    haskey(COMPARTMENTS, comp_key) || return nothing
    
    config = COMPARTMENTS[comp_key]
    p = nothing
    
    for (idx, (data, label)) in enumerate(zip(datasets, labels))
        if haskey(data, comp_name)
            obs = data[comp_name]
            color = _cycle(config.colors, idx)
            marker = _cycle(MARKERS, idx)
            yerror = _calc_errorbars(obs, show_errorbars)
            p = _scatter_data!(p, obs.times, obs.values, label; 
                             yerror=yerror, color=color, marker=marker, alpha=alpha)
        end
    end
    
    isnothing(p) ? nothing : plot!(p; title=config.title)
end

"""
Create plot for a single compartment with observed data and/or solutions.
"""
function _plot_compartment_combined(comp_name::String; 
                                   datasets=nothing, 
                                   solutions=nothing,
                                   data_labels=String[],
                                   solution_labels=String[],
                                   state_index::Int=1,
                                   show_errorbars::Bool=true,
                                   alpha::Float64=0.7)
    comp_key = Symbol(comp_name)
    haskey(COMPARTMENTS, comp_key) || return nothing
    
    config = COMPARTMENTS[comp_key]
    p = nothing
    
    # Plot solutions first (as lines)
    if !isnothing(solutions)
        for (idx, (sol, label)) in enumerate(zip(solutions, solution_labels))
            color = _cycle(config.colors, idx)
            p = _plot_line!(p, sol.t, sol[state_index, :], label; color=color)
        end
    end
    
    # Plot observed data (as scatter points)
    if !isnothing(datasets)
        offset = isnothing(solutions) ? 0 : length(solutions)
        for (idx, (data, label)) in enumerate(zip(datasets, data_labels))
            if haskey(data, comp_name)
                obs = data[comp_name]
                color = _cycle(config.colors, offset + idx)
                marker = _cycle(MARKERS, idx)
                yerror = _calc_errorbars(obs, show_errorbars)
                p = _scatter_data!(p, obs.times, obs.values, label; 
                                 yerror=yerror, color=color, marker=marker, alpha=alpha)
            end
        end
    end
    
    isnothing(p) ? nothing : plot!(p; title=config.title)
end

"""
    plot_cll(; observed=nothing, solutions=nothing, title="CLL Data", ...)

Unified plotting function for CLL data and/or model solutions.

# Arguments
- `observed`: Single `Dict{String, ObservedData}` or `Vector` of them (optional)
- `solutions`: Single solution or `Vector` of solutions (optional)
- `title::String`: Main plot title
- `data_labels::Vector{String}`: Labels for observed datasets
- `solution_labels::Vector{String}`: Labels for solutions
- `show_errorbars::Bool`: Whether to show error bars on observed data
- `alpha::Float64`: Transparency of data markers
- `tissue_index::Int`: State index for tissue compartment (default: 1)
- `blood_index::Int`: State index for blood compartment (default: 2)
- `layout`: Plot layout

At least one of `observed` or `solutions` must be provided.

# Examples
```julia
# Just observed data
observed = load_cll_data("dataset1.csv")
plot_cll(observed=observed, data_labels=["Patient A"])

# Just solutions
sol = simulate(model, (0.0, 500.0))
plot_cll(solutions=sol, solution_labels=["Model"])

# Both solutions and data
plot_cll(observed=observed, solutions=sol, 
         data_labels=["Data"], solution_labels=["Fit"])

# Multiple of each
plot_cll(observed=[obs1, obs2], solutions=[sol1, sol2],
         data_labels=["Patient A", "Patient B"],
         solution_labels=["Fast Model", "Slow Model"])
```
"""
function plot_cll(; observed=nothing,
                    solutions=nothing,
                    title::String="CLL Data",
                    data_labels::Vector{String}=String[],
                    solution_labels::Vector{String}=String[],
                    show_errorbars::Bool=true,
                    alpha::Float64=0.7,
                    tissue_index::Int=1,
                    blood_index::Int=2,
                    layout=(2,1))
    
    # Validate inputs
    isnothing(observed) && isnothing(solutions) && 
        throw(ArgumentError("At least one of 'observed' or 'solutions' must be provided"))
    
    # Normalize inputs to vectors
    datasets = isnothing(observed) ? nothing : 
               (observed isa Vector ? observed : [observed])
    sols = isnothing(solutions) ? nothing : 
           (solutions isa Vector ? solutions : [solutions])
    
    # Generate default labels if not provided
    final_data_labels = if !isnothing(datasets)
        isempty(data_labels) ? ["Data $i" for i in 1:length(datasets)] : data_labels
    else
        String[]
    end
    
    final_solution_labels = if !isnothing(sols)
        isempty(solution_labels) ? ["Model $i" for i in 1:length(sols)] : solution_labels
    else
        String[]
    end
    
    # Validate label counts
    !isnothing(datasets) && length(final_data_labels) != length(datasets) &&
        throw(ArgumentError("Number of data_labels must match number of observed datasets"))
    !isnothing(sols) && length(final_solution_labels) != length(sols) &&
        throw(ArgumentError("Number of solution_labels must match number of solutions"))
    
    # Create plots for each compartment
    compartments = ["tissue", "blood"]
    indices = [tissue_index, blood_index]
    
    plots = [_plot_compartment_combined(comp; 
                                       datasets=datasets,
                                       solutions=sols,
                                       data_labels=final_data_labels,
                                       solution_labels=final_solution_labels,
                                       state_index=idx,
                                       show_errorbars=show_errorbars,
                                       alpha=alpha)
             for (comp, idx) in zip(compartments, indices)]
    
    valid_plots = filter(!isnothing, plots)
    plot(valid_plots...; layout=layout, size=PLOT_DEFAULTS.size, plot_title=title)
end
