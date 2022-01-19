# Estimating arbitrary parameters for models

Provide a new parallel infrastructure based around dicts

New fields:

- `fixed`: pair of index and float that provide values to fix

New methods:

- `parameter_names`: positions of parameters
- `complete_parameters`: Complete a parameter vector. Dict will be turned into vector, vector will get indices replaced.
- `::Type{T} where {T <: AbstractModel}`: New constructors for fixing values.

# Workflow

```
mod = Model((0.0, 600.0), dat.times, dat.index, fixed=Dict(:ps=>0.95))
mod_starts = starts(mod) # return 5x100 array (partial)
objective(mod, euclidean, value, mod_starts[1,:]) # return single value
objective(mod, euclidean, value, Dict(...)) # return single value

```

# Progress

codes:

- `i`: implemented
- `o`: old version should work
- `t`: tested
- `x`: complete
- `n`: no changes

- [x] Constructors
- [x] `free_parameters(::AbstractModel)`
- [x] `free_parameter_count(::AbstractModel)`
- [x] starting value only of needed dimensionality
- [x] `parameter_array(::AbstractModel, ::Dict)`
- [x] `objective(::Array)`
- [x] `objective(::Dict)` -> goes to array version
- [x] `simulate(::Array)`
- [x] `simulate(::Dict)` -> goes to array version
- [x] `initial(::Array)` -> no change as should not do anything new?
- [x] `initial(::Dict)` -> goes to array version
- [ ] transformations only in appropriate subspace
