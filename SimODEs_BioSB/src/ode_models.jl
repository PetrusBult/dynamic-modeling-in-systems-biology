using OrdinaryDiffEq, StaticArrays

"""
    AbstractODEModel

Abstract base type for all ODE models.

# Interface Requirements

Concrete subtypes must implement:

## Required Fields
- `parameters::NamedTuple` - Model parameters with named values
- `initial_conditions::NamedTuple` - Initial state values with named variables

## Required Methods
- `ode_func!(model::ConcreteModel, du, u, p, t)` - ODE right-hand side function
  - `du`: Derivative vector (output)
  - `u`: State vector (input)
  - `p`: Parameters (NamedTuple)
  - `t`: Time
  - Must return `nothing`
"""
abstract type AbstractODEModel end

struct WodartzModel <: AbstractODEModel
    parameters::NamedTuple
    initial_conditions::NamedTuple
end

function ode_func!(model::WodartzModel, du, u, p, t)
  @inbounds begin
    # Unpack parameters
    m = p[1] # redistribution from tissue to blood
    d1 = p[2] # death rate in the tissue
    d2 = p[3] # death rate in the blood
    c = p[4] # accounts for observation that lymphocytes in the tissue do not reach complete clearance

    du[1] = -m * u[1] - d1 * (u[1] - c) # Tissue compartment
    du[2] = m * u[1] - d2 * u[2]         # Blood compartment
  end
  nothing
end

# Default constructor
WodartzModel() = WodartzModel(
  (m=0.1, d1=0.05, d2=0.2, c=1.0), # default parameters
  (tissue=10.0, blood=5.0)         # default initial conditions
)
