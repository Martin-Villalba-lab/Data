# Methods for this:
# - plot (plot data)
# - times (time for value)
# - values (value for value)
# - field (field for value)
# - weight (weight for value)
struct ModelData
    x::Vector{Float64} # value
    t::Vector{Float64} # time
    f::Vector{Int64} # field
    w::Vector{Float64} # weight
    #fieldnames::Vector{String}
    ModelData(x, t, f, w) = length(t) == length(x) == length(f) == length(w) ? new(x, t, f, w) : @error "Mismatched lengths"
end

weights(x::ModelData) = x.w
times(x::ModelData) = x.t
fields(x::ModelData) = x.f
vals(x::ModelData) = x.x

function DataFrames.DataFrame(x::ModelData) 
    DataFrame(:value => vals(x),
              :time => times(x),
              :fields => fields(x),
              :weight => weights(x))
end

# Most important constructor.
function ModelData(d::DataFrame)
    fields = names(d)
    if !("time" in fields)
        @error "Missing column time"
    end
    if !("value" in fields)
        @error "Missing column value"
    end
    if "weight" in fields
        weight = d.weight
    else
        weight = [1 for _ in 1:size(d, 1)]
    end
    if "field" in fields
        field = d.field
    else
        field = [1 for _ in 1:size(d, 1)]
    end
    ModelData(d.value, d.time, field, weight)
end

"""AbstractModel

`AbstractModel`s consist of the following functions to make modelling easier:

- θ -> D, called `simulate`. Takes parameters and returns model output
- D -> D*, called `link`. Takes model output and returns the data observed

To create a model, implement:

- `parameter_names`
- `simulate`

Optionally:

- `link` (defaults to identity)
- `output_names` (not technically used)
- `link_names` (not technically used)
"""
abstract type AbstractModel end

output_names(t::AbstractModel) = @error "Not implemented"
link_names(t::AbstractModel) = @error "Not implemented"

"""simulate(t::AbstractModel, x)
simulate(t::AbstractModel, x::AbstractVector)
simulate(t::AbstractModel)::Function

Simulate a model using parameters provided in `x`
"""
simulate(t::AbstractModel, x::AbstractVector) = @error "Not implemented"# = simulate(t, x...)
simulate(t::AbstractModel, x::AbstractDict{Symbol, N}) where {N <: Number} = simulate(t, parameter_array(t, x))
simulate(t::AbstractModel) = x -> simulate(t, x)

"""link(t::AbstractModel, x::AbstractArray)
link(t::AbstractModel)::Function

Takes model outputs and applies transformations to it to make it appear like
the data that the user provided.
Defaults to `identity`, overload at your leisure.

Provides a curried version, `link(::AbstractModel)::Function`
"""
link(t::AbstractModel, x::AbstractArray) = identity(x)
link(t::AbstractModel) = x -> link(t, x)

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

"""parameter_dict(t::AbstractModel, x::AbstractArray)

Create a dictionary with parameters
"""
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

"""bounds(t::AbstractModel)::Vector{Tuple{Number, Number}}

Bounds for the models optimiser space
"""
bounds(t::AbstractModel) = @error "Not implemented"
bounds(t::AbstractModel, parameter::Integer) = bounds(t)[parameter]
bounds(t::AbstractModel, parameter::Symbol) = bounds(t)[parameter_index(t, parameter)]
lower_bound(t::AbstractModel) = map(param -> bounds(t, param)[1], free_parameters(t))
upper_bound(t::AbstractModel) = map(param -> bounds(t, param)[2], free_parameters(t))


"""ODEModel

Extends AbstractModel by providing a `simulate`.
`simulate` also returns a function that handles timepoints unless given some timepoints.

Additionally provides fields:

- `tspan`: timespan to solve in

Additionally provides methods:

- `initial`
- `ratefun`

To implement an `ODEModel`, implement:

- `initial`
- `ratefun`
- `parameter_names`
"""
abstract type ODEModel <: AbstractModel end

# Empty model
function (::Type{T})(tspan::Tuple{Float64, Float64}) where {T <: ODEModel}
    T(tspan, Vector{Pair{Int64, Float64}}())
end
# Allow symbolic parameter fixation.
function (::Type{T})(tspan::Tuple{Float64, Float64}, fixed::AbstractDict{Symbol, N}) where {T <: ODEModel, N <: Number}
    names = parameter_names(T)
    fixed = filter(x -> x.first in names, fixed) # only keep keys that are in the names for this type
    fixed = [findfirst(key .== names) => value for (key, value) in fixed]
    T(tspan,  fixed)
end

initial(::ODEModel, x::AbstractVector) = @error "Not implemented"
ratefun(::ODEModel) = @error "Not implemented"

function simulate(t::ODEModel, x::AbstractVector)
    x = length(x) != parameter_count(t) ? parameter_array(t, x) : x
    u₀ = initial(t, x)
    rates! = ratefun(t)
    problem = ODEProblem(rates!, u₀, t.tspan, x)
    solution = solve(problem)#, alg_hints=[:stiff])
    #times = LinRange(t.tspan[1], t.tspan[2], steps)
    times -> collect(hcat(solution(times).u...)')
end
function simulate(t::ODEModel, x::AbstractVector, times::AbstractVector)
    simulate(t, x)(times)
end
simulate(t::ODEModel, x::AbstractDict{Symbol, N}, times::AbstractVector) where {N <: Number} = simulate(t, parameter_array(t, x), times)

#simulate_link(t::ODEModel, x::AbstractVector, times::AbstractVector) = simulate(t, x, times) |> link(t)

struct ModelObjective <: Function
    d::ModelData
    m::AbstractModel
end


vals(m::ModelObjective) = vals(m.d)
fields(m::ModelObjective) = fields(m.d)
times(m::ModelObjective) = times(m.d)
weights(m::ModelObjective) = weights(m.d)
model(m::ModelObjective) = m.m

function simulate(m::ModelObjective, x::AbstractArray)
    simulate(model(m), x, times(m)) |> link(model(m))
end

function (m::ModelObjective)(x; ε=1e-10, pm=nothing)
    x = parameter_array(model(m), x)
    b = bounds(model(m))
    fixed_indexes = [f.first for f in model(m).fixed]
    x = [i in fixed_indexes ? xi : min(max(xi, bi[1]+ε), bi[2]-ε) for (i, (bi, xi)) in enumerate(zip(b, x))]
    sim = [s[i] for (i, s) in zip(fields(m), eachrow(simulate(m, x)))]
    w = weuclidean(sim, vals(m), weights(m))
    if !isnothing(pm)
        shows = collect(zip(parameter_names(model(m)), x))
        push!(shows, (:dist, w))
        ProgressMeter.next!(pm; showvalues=shows)
    end
    w
end

function starts(mo::ModelObjective; n=10, ε=1e-20, iters=200)
    mod = model(mo)
    param_bounds = map(free_parameters(mod)) do param
        bounds(mod, param)
    end
    param_values, _ = LHCoptim(n, length(param_bounds), iters)
    scaleLHC(param_values, [(b[1]+ε, b[2]-ε) for b in param_bounds])
end

function Optim.optimize(mo::ModelObjective, s::Vector)
    prog = ProgressUnknown("Iterations:")
    mod = model(mo)
    o = Optim.optimize(x -> mo(x; pm=prog), lower_bound(mod), upper_bound(mod), s,
                       Fminbox(NelderMead()), Optim.Options(f_abstol=1e-2, time_limit=120))
    ProgressMeter.finish!(prog)
    o
end

function Optim.optimize(mo::ModelObjective, s::Matrix)
    map(eachrow(s)) do start
        Optim.optimize(mo, collect(start))
    end
end

function Optim.optimize(mo::ModelObjective; n=10, ε=1e-10)
    Optim.optimize(mo, starts(mo; n=n, ε=ε))
end

struct ModelFit
    obj::ModelObjective # objective function used
    fits::Array # all fits
    best#::Optim.OptimizationResults # best fit
end

function ModelFit(mo::ModelObjective; n=10, ε=1e-10)
    opts = Optim.optimize(mo, n=n, ε=ε)
    best = opts[minimum(minimum.(opts)) .== minimum.(opts)][1].minimizer
    ModelFit(mo, opts, best)
end

function parameter_array(mf::ModelFit)
    parameter_array(model(mf.obj), mf.best)
end

function parameter_dict(mf::ModelFit)
    parameter_dict(model(mf.obj), mf.best)
end

function simulate(mf::ModelFit)
    fun = mf.obj
    times = LinRange(fun.m.tspan..., 100) |> collect
    simulate(mf.obj.m, mf.best)(times) |> link(mf.obj.m)
end

# Now we should probably just provide some extra convenience methods

# Plots

@recipe function f(md::ModelData)
    panels = fields(md) |> unique
    d = DataFrame(md)
    pans = [(label=i, blank=false) for i in 1:length(panels)]
    layout --> reshape(pans, 1, length(pans))
    markercolor --> :black
    seriestype := :scatter
    xlabel --> "Time"
    for panel in panels
        x = d[d.fields .== panel, "time"]
        y = d[d.fields .== panel, "value"]
        @series begin
            subplot := panel
            label --> ""
            seriestype := :scatter
            (x, y)
        end
    end
end

@recipe function f(mf::ModelFit)
    fun = mf.obj
    data = DataFrame(fun.d)
    ls = link_styles(mf.obj.m)
    optim = mf.best
    sim = simulate(mf)
    times = LinRange(fun.m.tspan..., 100) |> collect
    @series begin
        mf.obj.d
    end
    for i in 1:size(sim, 2)
        @series begin
            datavalues = data[data.fields .== i, "value"]
            lims = (minimum(datavalues), maximum(datavalues))
            yscale --> ls[i]
            limdiff = lims[2] - lims[1]
            #ylimits --> (lims[1] - 0.1*limdiff, lims[2] + 0.1*limdiff)
            ylabel --> link_names(mf.obj.m)[i]
            label --> ""
            width --> 2
            linecolor --> "red"
            (times, sim[:,i])
        end
    end
end

residuals(x::ModelFit) = x.obj(x.best)
sample_size(x::ModelFit) = length(x.obj.d.x)
free_parameter_count(x::ModelFit) = free_parameter_count(x.obj.m)

function AIC(x::ModelFit)
    r = residuals(x)
    n = sample_size(x)
    k = free_parameter_count(x)
    n * log(r/n)+2*k+(2*k^2+2*k) / (n-k-1)
end

#@recipe function f(::Type{ModelData}, x::ModelData)
#    panels = fields(x) |> unique
#    d = DataFrame(x)
#    layout := [(label=i, blank=false) for i in 1:length(panels)]
#    legend := :false
#    seriestype := :scatter
#    for panel in panels
#        x = d[d.fields .== panel,"time"]
#        y = d[d.fields .== panel,"value"]
#        @series begin
#            subplot := panel
#            seriestype := :scatter
#            (x, y)
#        end
#    end
#end

exponential_decay(x₀, β, x) = x₀ * exp(-β*x)
flattening_curve(x₀, β, x) = 0.5 * (1 + exp(-β*x) * (2*x₀-1))
hill(k, n, x) = 1/(1+(k/x)^n)
hill(k, n) = x -> hill(k, n, x)
