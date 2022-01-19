using Revise

using DifferentialEquations
using Optim
using Statistics
using Distances
using LatinHypercubeSampling

using CSV
using DataFrames
using DataFramesMeta
using IterTools

using Plots
gr()

using ProgressMeter

ProgressMeter.ijulia_behavior(:clear)

includet("models/base.jl")
includet("models/nsc_kalamakis.jl")

tmin = 0.0
tmax = 700.0
tspan = (tmin, tmax)
pₛ = log(2)*(24/17.5)#0.95

data = map(["wt", "ifnko"]) do geno
    active = @linq CSV.read("data/data_$(geno)_active.csv", DataFrame) |>
        transform(:name="active") |>
        select(:genotype=:genotype, :time=:age, :name=:name, :value=:active, :weight=:weight)
    counts = @linq CSV.read("data/data_$(geno)_counts.csv", DataFrame) |>
        transform(:name="total") |>
        select(:genotype=:genotype, :time=:age, :name=:name, :value=:count, :weight=:weight)
    vcat([active, counts]...)
end
data = vcat(data...)
data = @linq data |> transform(:field=[n .== "active" ? 2 : 1 for n in :name])

genotypes = Dict(:wt => ModelData(@subset(data, :genotype .== "wt")),
                 :ko => ModelData(@subset(data, :genotype .== "ifnko")))

plot([plot(p, title=k) for (k, p) in genotypes]..., layout=@layout([a; b]), size=(600, 600))

models = Dict(
    :both       => KalamakisFull(tspan, Dict(:pₛ => pₛ)),
    :self       => KalamakisFull(tspan, Dict(:pₛ => pₛ, :βr => 0.0)),
    :activation => KalamakisFull(tspan, Dict(:pₛ => pₛ, :βb => 0.0)),
    :noageing   => KalamakisFull(tspan, Dict(:pₛ => pₛ, :βb => 0.0, :βr => 0.0)),
)

results = map(IterTools.product(keys(genotypes), keys(models))) do (genotype_name, model_name)
    mo = ModelObjective(genotypes[genotype_name], models[model_name])
    (genotype_name, model_name) => ModelFit(mo, n=28)
end

results_dict = Dict(results[:]...)

plot(map(x -> plot(x.second, title="$(x.first[1]) $(x.first[2])"), results)..., layout=(4, 2), size=(1000, 1000))

Dict(k => parameter_dict(x) for (k, x) in results_dict)

params = DataFrame(collect(hcat([parameter_array(x) for x in values(results_dict)]...)'), parameter_names(KalamakisFull))
params[!, :genotype] = [k[1] for k in keys(results_dict)]
params[!, :model] = [k[2] for k in keys(results_dict)]
params[!, :r_w] = [residuals(results_dict[k]) for k in keys(results_dict)]
params[!, :aicc] = [AIC(results_dict[k]) for k in keys(results_dict)]
order = vcat([:genotype, :model, :r_w, :aicc, :Δaicc], parameter_names(KalamakisFull))
params = @linq params |>
    orderby(:genotype, :aicc) |>
    DataFramesMeta.groupby(:genotype) |>
    transform(:Δaicc=:aicc .- :aicc[1]) |>
    orderby(:genotype, :Δaicc)
ENV["COLUMNS"] = 160
params[!,order]

r = results_dict[:wt, :noageing]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:ko, :noageing]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:wt, :activation]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:ko, :activation]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:wt, :self]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:ko, :self]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:wt, :both]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

r = results_dict[:ko, :both]
parms = parameter_dict(r)
for k  in parameter_names(KalamakisFull)
    println("$k: $(parms[k])")
end
println("r_w: $(r.obj(r.best))")
plot(r, xlabel="Time (days)")

pkgs = IOBuffer()
loaded = Base.loaded_modules_array()
using Pkg; Pkg.status(io=pkgs)
seek(pkgs, 0)
pkgs = split(String(read(pkgs)), "\n")
for pkg in pkgs
    if any([contains(pkg, "$l") for l in loaded])
        println(pkg)
    end
end
