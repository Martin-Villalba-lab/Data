# Transformations for spaces
logistic(x₁, x₂, x) = inftymap10⁻¹(x₁, x₂, 1/(1 + exp(-x - x₁)))
logistic(x₁, x₂) = x -> logistic(x₁, x₂, x)
inftymap10(x₁, x₂, x) = (x - x₁) / (x₂ - x₁)
inftymap10(x₁, x₂) = x -> inftymap10(x₁, x₂, x)
inftymap10⁻¹(x₁, x₂, x) = x *(x₂ - x₁) + x₁
inftymap10⁻¹(x₁, x₂) = x -> inftymap10⁻¹(x₁, x₂, x)
logit(x) = log(x / (1 - x))
logit⁻¹(x) = logistic(x, 0, 1)
xlogit(x₁, x₂, x) = (logit ∘ inftymap10)(x₁, x₂, x)
xlogit(x₁, x₂) = x -> xlogit(x₁, x₂, x)
xlogit⁻¹ = logistic

"""AbstractModel

`AbstractModel`s consist of the following functions to make modelling easier:

- θ -> D, called `simulate`. Takes parameters and returns model output
- D -> D*, called `link`. Takes model output and returns the data observed
- θ* -> θ, `transform` transforms from optimiser space to parameter space (typically unbounded ℝⁿ to some subset) 
- θ -> θ*, `transform⁻¹` inverse of the above
- θ -> r, `objective` function
"""
abstract type AbstractModel end

"""ODEModel

An extension of `AbstractModel` for dynamical systems.
Introduces extra fields:

- `tspan`: The timespan of the solutions to create.
- `times`: The times that fitted data is from.
- `values`: Specfies which of the linked outputs the data belongs to.

Provides extra methods:

- `dense`
- `initial`
- `ratefun`
"""
abstract type ODEModel <: AbstractModel end

# Generic ODEMOdel constructors
# Allow very free model construction
function (::Type{T})(tspan::Tuple{Float64, Float64}, times::Vector{Float64}, values::Vector{Int64}) where {T <: ODEModel}
    T(tspan, times, values, Vector{Pair{Int64, Float64}}())
end

# Allow symbol parameter fixation
function (::Type{T})(tspan::Tuple{Float64, Float64}, times::Vector{Float64}, values::Vector{Int64}, fixed::AbstractDict{Symbol, N}) where {T <: ODEModel, N <: Number}
    names = parameter_names(T)
    fixed = filter(x -> x.first in names, fixed) # only keep keys that are in the names for this type
    fixed = [findfirst(key .== names) => value for (key, value) in fixed]
    T(tspan, times, values, fixed)
end



# Parameter thrashing

"""simulate(t::AbstractModel, x)
simulate(t::AbstractModel, x::AbstractVector)
simulate(t::AbstractModel)::Function

Simulate a model using parameters provided in `x`
"""
simulate(t::AbstractModel, x::AbstractVector) = @error "Not implemented"# = simulate(t, x...)
simulate(t::AbstractModel, x::AbstractDict{Symbol, N}) where {N <: Number} = simulate(t, parameter_array(t, x))
simulate(t::AbstractModel) = x -> simulate(t, x)

function simulate(t::ODEModel, x::AbstractVector)
    u₀ = initial(t, x)
    rates! = ratefun(t)
    problem = ODEProblem(rates!, u₀, t.tspan, x)
    solution = solve(problem)
    collect(hcat(solution(t.times).u...)')
end
ratefun(t::ODEModel) = @error "Not implemented"

# Methods for AbstractModels
"""link(t::AbstractModel, x::AbstractArray)
link(t::AbstractModel)::Function

Takes model outputs and applies transformations to it to make it appear like
the data that the user provided.
Defaults to `identity`, overload at your leisure.

Provides a curried version, `link(::AbstractModel)::Function`
"""
link(t::AbstractModel, x::AbstractArray) = identity(x)
link(t::AbstractModel) = x -> link(t, x)

"""transform(t::AbstractModel, x::AbstractVector)::AbstractVector
transform(t::AbstractModel, n::Integer, x::Number)::Number
transform(t::AbstractModel)::Function

Transform parametervector (or scalar parameter) to optimiser space.

"""
function transform(t::AbstractModel, x::AbstractVector)
    names = free_parameters(t)
    these_bounds = [bounds(t, n) for n in names]
    [logistic(b[1], b[2], y) for (y, b) in zip(x, these_bounds)]
end
transform(t::AbstractModel, n::Integer, x::Number) = logistic(bounds(t)[n][1], bounds(t)[n][2], x)
transform(t::AbstractModel) = x -> transform(t, x)
transform(t::AbstractModel, n::Integer) = x -> transform(t, n, x)

"""transform⁻¹(t::AbstractModel, x::AbstractVector)::AbstractVector
transform⁻¹(t::AbstractModel, n::Integer, x::Number)::Number
transform⁻¹(t::AbstractModel)::Function

Transform optimiser space parameter vector (or scalar parameter) to normal space.

"""
function transform⁻¹(t::AbstractModel, x::AbstractVector)
    names = free_parameters(t)
    these_bounds = [bounds(t, n) for n in names]
    [xlogit(b[1], b[2], y) for (y, b) in zip(x, these_bounds)]
end
transform⁻¹(t::AbstractModel, n::Integer, x::Number) = xlogit(bounds(t)[n][1], bounds(t)[n][2], x)
transform⁻¹(t::AbstractModel) = x -> transform⁻¹(t, x)
transform⁻¹(t::AbstractModel, n::Integer) = x -> transform(t, n, x)

"""bounds(t::AbstractModel)::Vector{Tuple{Number, Number}}

Bounds for the models optimiser space
"""
bounds(t::AbstractModel) = @error "Not implemented"
bounds(t::AbstractModel, parameter::Integer) = bounds(t)[parameter]
bounds(t::AbstractModel, parameter::Symbol) = bounds(t)[parameter_index(t, parameter)]

"""parameter_names(t::AbstractModel)

Returns the names for the parameter vector(s)
"""
parameter_names(t::Type{AbstractModel}) = @error "Not implemented"
parameter_names(t::AbstractModel) = parameter_names(typeof(t))

"""free_parameters(t::AbstractModel)

Return vector of fittable (i.e. non-fixed) parameter Symbols for model `t`
"""
function free_parameters(t::AbstractModel)
    parameters_should = parameter_names(t)
    parameters_have = [parameters_should[fix.first] for fix in t.fixed]
    setdiff(parameters_should, parameters_have)
end

"""free_parameter_count(t::AbstractModel)

How many parameters are free in model `t`
"""
function free_parameter_count(t::AbstractModel)
    length(free_parameters(t))
end

"""parameter_count(t::AbstractModel)

How many parameters are present in model `t` (total)
"""
function parameter_count(t::AbstractModel)
    length(parameter_names(t))
end

"""parameter_index(t::AbstractModel, name::Symbol)

Return index of parameter `name` in model `t`
"""
function parameter_index(t::AbstractModel, name::Symbol)
    findfirst(parameter_names(t) .== name)
end

"""parameter_array(t::AbstractModel, d::Dict{Symbol, Number})
parameter_array(t::AbstractModel, x::Array)

Create a parameter array (full) for model `t` from either a dict `d` or an array.
"""

function parameter_array(t::AbstractModel, d::AbstractDict{Symbol, N}; eltype=Float64) where {N <: Number}
    n = parameter_count(t)
    parameters = zeros(eltype, n)
    for (name, value) in d
        parameters[parameter_index(t, name)] = value
    end
    for (index, value) in t.fixed
        parameters[index] = value
    end
    parameters
end
function parameter_array(t::AbstractModel, x::AbstractArray; eltype=Float64)
    n = parameter_count(t)
    parameters = zeros(eltype, n)
    skipindices = map(x -> x.first, t.fixed)
    xi = 1
    for i=1:n
        if i in skipindices
            continue
        end
        parameters[i] = x[xi]
        xi +=1
    end
    for (index, value) in t.fixed
        parameters[index] = value
    end
    parameters
end
function parameter_dict(t::AbstractModel, x::AbstractArray)
    free = Dict(
         k => v
         for (v, k) in zip(x, free_parameters(t))
    )
    fix = Dict(
        parameter_names(t)[x.first] => x.second
        for x in t.fixed
    )
    merge(free, fix)
end
function parameter_dict(t::AbstractModel, x::AbstractDict)
    fix = Dict(
        parameter_names(t)[x.first] => x.second
        for x in t.fixed
    )
    merge(x, fix)
end



"""objective(t::AbstractModel, d::Function, y::AbstractArray, x::AbstractArray)
objective(t::AbstractModel, d::Function, y::AbstractArray)::Function
objective(t::AbstractModel, d::Function)::Function
objective(t::ODEModel, d::Function, y::AbstractArray, x::AbstractArray)
objective(t::ODEModel, d::Function, y::AbstractArray, x::AbstractDict{Symbol, Number})

objective function fo a model.

- `t` the model
- `d` should be the distance function to use
- `x` a set of parameters
- `y` an array of data
"""
#objective(t::AbstractModel, d::Function, y::AbstractArray, x::AbstractVector) = d(transform(t, x) |> simulate(t) |> link(t), y)
objective(t::AbstractModel, d::Function, y::AbstractArray) = x -> objective(t, d, y, x)
objective(t::AbstractModel, d::Function) = y -> x -> objective(t, d, y, x)
function objective(t::AbstractModel, d::Function, y::AbstractArray, x::AbstractArray)
    result = parameter_array(t, x) |> simulate(t) |> link(t)
    r = [x[tt] for (x, tt) in zip(eachrow(result), t.values)]
    d(r, y)
end
# TODO: think if this is actually needed
objective(t::ODEModel, d::Function, y::AbstractArray, x::AbstractDict{Symbol, N}) where {N <: Number} = objective(t, d, y, parameter_array(t, x))

trans_objective(t::AbstractModel, d::Function, y::AbstractArray) = objective(t, d, y) ∘ transform(t)

#struct OptimisationResult
#    optims::Array{Any}
#    minimiser::AbstractVector
#    minimum::Number
#    minimisers::AbstractMatrix
#    minimae::AbstractVector{Number}
#    minimum_index::Integer
#end

"""optimise(t::AbstractModel, d::Function, x::AbstractArray, start::AbstractVector)
optimise(t::AbstractModel, d::Function, x::AbstractArray, start::AbstractMatrix)

- t: Model
- d: Distance function
- x: Data vector
- start: starting parameters. Either a matrix (rows for each starting vector) or a single starting vector
"""
function optimise(t::AbstractModel, d::Function, x::AbstractArray, start::AbstractVector)
    ϕ = trans_objective(t, d, x)
    options = Optim.Options(store_trace=false, iterations=10_000)
    optimize(ϕ, transform⁻¹(t, start), options)
end

function optimise(t::AbstractModel, d::Function, x::AbstractArray, start::AbstractMatrix)
    map(eachrow(start)) do s
        optimise(t, d, x, s)
    end
end

"""starts(t::AbstractModel, n=100, ε=1e-10, iters=200)

Sample latinhypercube starting values from the parameter space.
Returns actual parameter space values.

Parameters:

- ε: amount to push values away from the edge (avoids numerical issues with transformations)
- n: Samples to generate
- iters: Number of latin hypercube optimisation iterations
"""
function starts(t::AbstractModel; n=100, ε=1e-10, iters=200)
    param_bounds = map(free_parameters(t)) do param
        bounds(t, param)
    end
    param_values, _ = LHCoptim(n, length(param_bounds), iters)
    scaleLHC(param_values, [(b[1]+ε, b[2]-ε) for b in param_bounds])
end

# Methods specific to ODEModels
"""initial(t::ODEModel, x::AbstractVector)

Function that creates an initial value from the parameters in `x`.
"""
initial(t::ODEModel, x::AbstractVector) = @error "Not implemented"
initial(t::ODEModel, x::AbstractDict{Symbol, N}) where {N <: Number}= initial(t, parameter_array(t, x))

"""dense(t::ODEModel)

Creates a copy of `t` that has a more dense `times` field (as sppecified by the `tspan` field)
"""
function dense(t::ODEModel)
    fnames = fieldnames(t |> typeof) |> collect
    fields = map(x -> getfield(t, x), fnames)
    fields[findfirst(fnames .== :times)] = collect(t.tspan[1]:t.tspan[2])
    fields[findfirst(fnames .== :values)] = repeat([1], length(t.values))
    typeof(t)(fields...)
end

output_names(t::AbstractModel) = @error "Not implemented"
link_names(t::AbstractModel) = @error "Not implemented"

function AIC(t::AbstractModel, r::Number, n::Number)
    k = free_parameter_count(t)
    n * log(r/n)+2*k+(2*k^2+2*k) / (n-k-1)
end

#function AIC(t::AbstractModel, x::AbstractVector, d::Function,  y::AbstractVector)
#end

# Parameter functions
exponential_decay(x₀, β, x) = x₀ * exp(-β*x)
flattening_curve(x₀, β, x) = 0.5 * (1 + exp(-β*x) * (2*x₀-1))
double_flattening_curve(x₀, β, x) = 2 * flattening_curve(x₀/2, β, x)
exponential_decay(x₀, β) = x -> exponential_decay(x₀, β, x)
flattening_curve(x₀, β) = x -> flattening_curve(x₀, β, x)
double_flattening_curve(x₀, β) = x -> double_flattening_curve(x₀, β, x)
