using Revise

using LatinHypercubeSampling
using DifferentialEquations
using Optim
using Statistics
using Distances
using Distributions
using Turing

using CSV
using DataFrames
using DataFramesMeta
using Chain: @chain
using Pipe: @pipe
using DataStructures

using Plots
using StatsPlots
using Interact
#pyplot()

includet("models/base.jl")

data = map(["wt", "ifnko"]) do geno
    active = @linq CSV.read("data_old/data_$(geno)_active.csv", DataFrame) |>
        transform(name="active") |>
        select(genotype=:genotype, age=:age, name=:name, value=:active, weight=:weight)
    counts = @linq CSV.read("data_old/data_$(geno)_counts.csv", DataFrame) |>
        transform(name="total") |>
        select(genotype=:genotype, age=:age, name=:name, value=:count, weight=:weight)
    vcat([active, counts]...)
end
data = vcat(data...)
data = @linq data |> transform(link=[n .== "active" ? 2 : 1 for n in :name])

sim_ifnko_both = CSV.read("data_old/solution_ifnko_self_renewal_and_quiescence.csv", DataFrame)
sim_wt_both = CSV.read("data_old/solution_wt_self_renewal_and_quiescence.csv", DataFrame)
sims = vcat(@transform(sim_wt_both, genotype="wt"), @transform(sim_ifnko_both, genotype="ko"))

p = plot(xlab="NSC", ylab="b", xflip=false)
for group in groupby(sims, :genotype)
    @df group plot!(p, :counts, :b, label=first(group.genotype))
end
p

p = plot(xlab="NSC", ylab="b", xflip=false, xscale=:log10)
for group in groupby(sims, :genotype)
    @df group plot!(p, :counts, :b, label=first(group.genotype))
end
p

counts = @where(sims, :genotype .== "ko").counts
b =  @where(sims, :genotype .== "ko").b 
opt = optimize(x -> euclidean(x[1].+x[2]*log10.(counts), b), [1.0, 1.0])

opt.minimizer

p = plot(xlab="NSC", ylab="b", xflip=false, xscale=:log10)
for group in groupby(sims, :genotype)
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
plot!(x -> opt.minimizer[1]+(opt.minimizer[2]*log10(x)), lab="ex", ls=:dash, lw=2)
p

hill(kd, n, x) = (x^n)/(kd+x^n)
hill(kd, n) = x -> hill(kd, n, x)

hill(ka, n, x) = 1/(1+(ka/x)^n)
hill(ka, n) = x -> hill(ka, n, x)

points = 10.0 .^ (-3:0.1:5)
plot(xscale=:log10, xlim=(0.001, 1000))
plot!(points, hill(.001, 1), lab="")
plot!(points, hill(.01, 1), lab="")
plot!(points, hill(.1, 1), lab="")
plot!(points, hill(1, 1), lab="")
plot!(points, hill(10, 1), lab="")
plot!(points, hill(100, 1), lab="")
plot!(points, hill(1000, 1), lab="")

points = 10.0 .^ (-3:0.1:5)
plot(xscale=:log10, xlim=(0.001, 1000))
plot!(points, hill(1, 1/4), lab="")
plot!(points, hill(1, 1/2), lab="")
plot!(points, hill(1, 1), lab="")
plot!(points, hill(1, 2), lab="")
plot!(points, hill(1, 4), lab="")

p = plot(xlab="NSC", ylab="b", xflip=false)
for group in groupby(sims, :genotype)
    @df group plot!(p, :counts, :b, label=first(group.genotype))
end
p

plot(points, x -> 1-hill(1, 1, x), lab="", xscale=:log10)

p = plot(points, x -> 1-hill(150, 0.05, x), lab="", xscale=:log10)
for group in groupby(sims, :genotype)
    @df group plot!(p, :counts, :b, label=first(group.genotype))
end
p

counts = @where(sims, :genotype .== "ko").counts
b =  @where(sims, :genotype .== "ko").b 
koopt = optimize(x -> euclidean(1 .- hill.(x[1], x[2], counts), b), [1.0, 1.0])

counts = @where(sims, :genotype .== "wt").counts
b =  @where(sims, :genotype .== "wt").b 
wtopt = optimize(x -> euclidean(1 .- hill.(x[1], x[2], counts), b), [1.0, 1.0])

hcat(koopt.minimizer, wtopt.minimizer)

points = 10.0 .^ (-1:0.1:4)
p = plot(xscale=:log10,)
plot!(p, points, x -> 1-hill(koopt.minimizer[1], koopt.minimizer[2], x), lab="", lc=:gray, ls=:dash)
plot!(p, points, x -> 1-hill(wtopt.minimizer[1], wtopt.minimizer[2], x), lab="", lc=:gray, ls=:dash)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2, lc=i)
end
p

points = 10.0 .^ (2:0.1:3.5)
p = plot(xscale=:log10,)
plot!(p, points, x -> 1-hill(koopt.minimizer[1], koopt.minimizer[2], x), lab="", lc=:gray, ls=:dash)
plot!(p, points, x -> 1-hill(wtopt.minimizer[1], wtopt.minimizer[2], x), lab="", lc=:gray, ls=:dash)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2, lc=i)
end
p

struct PopSelf <: ODEModel
    tspan::Tuple{Float64, Float64} 
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}} # pointer to thing to fix and value
    PopSelf(tspan, times, values,
                  fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end

function ratefun(model::PopSelf)
    function(du, u, p, t)
        _, r₀, βr, ba, bb, pₛ = p
        Q = u[1]; A = u[2]
        b = 1-hill(ba, bb, Q+A)
        du[1] = dQ = -exponential_decay(r₀, βr, t) * Q + 2 * b * pₛ * A
        du[2] = dA =  exponential_decay(r₀, βr, t) * Q - pₛ * A
    end
end

function initial(t::PopSelf, x::AbstractVector)
    nsc₀, r₀, _, ba, bb, pₛ = x
    b₀ = 1-hill(ba, bb, nsc₀)
    ratio = sqrt(((pₛ - r₀)/(2*r₀))^2 + (2*b₀*pₛ) / r₀)
    nsc₀ .* [1-1/(ratio+1), 1/(ratio+1)]
end

link(t::PopSelf, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]))

parameter_names(t::Type{PopSelf}) = [:nsc₀, :r₀, :βr, :ba, :bb, :pₛ]

bounds(t::PopSelf) = [(100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (10.0, 1000.0), (0.005, 2.0), (0.0, 1.0)]

output_names(t::PopSelf) = ["qNSC", "aNSC"]
link_names(t::PopSelf) = ["Total NSC", "Fraction active NSC"]

data_wt = @where(data, :genotype .== "wt")
model_wt = PopSelf((0.0, 700.0), Float64.(data_wt.age), data_wt.link, Dict(:pₛ => 0.95))

model_starts = starts(model_wt, n=28*5)

model_wt_opt = optimise(model_wt, (x, y) -> weuclidean(x, y, data_wt.weight), data_wt.value, model_starts);

minimae = minimum.(model_wt_opt)
min_index = argmin(minimae)
#scatter(sort(minimae), lab="", mc=:black)
scatter(minimae, lab="", mc=:black)
scatter!([min_index], [minimum(model_wt_opt[min_index])], lab="", mc=:red)

best_wt_opt = model_wt_opt[argmin(minimum.(model_wt_opt))]

minimum(best_wt_opt)

model_wt_plot = dense(model_wt)
param_wt = parameter_dict(model_wt,transform(model_wt, best_wt_opt.minimizer))

plotlyjs()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylab="Fraction active", xlab="Time (days)")
for opt in filter(x -> minimum(x) < 7.5, model_wt_opt)
    sim = parameter_array(model_wt_plot, transform(model_wt_plot, opt.minimizer)) |> simulate(model_wt_plot) |> link(model_wt_plot)
    plot!(p1, model_wt_plot.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p2, model_wt_plot.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
end
#best_i = 2
#sim = parameter_array(model_wt_plot, transform(model_wt_plot, model_wt_opt[best_i].minimizer)) |> simulate(model_wt_plot) |> link(model_wt_plot)
#plot!(p1, model_wt_plot.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model_wt_plot.times, sim[:,2], lab="", lc=:green, lw=2)
sim = param_wt |> simulate(model_wt_plot) |> link(model_wt_plot)
plot!(p1, model_wt_plot.times, sim[:,1], lab="", lc=:red, lw=2)
plot!(p2, model_wt_plot.times, sim[:,2], lab="", lc=:red, lw=2)
@df @where(data_wt, :link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data_wt, :link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2)

ba = param_wt[:ba]
bb = param_wt[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, sim[:,1], b, lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
p

data_ko = @where(data, :genotype .== "ifnko")
model_ko = PopSelf((0.0, 700.0), Float64.(data_ko.age), data_ko.link, Dict(:pₛ => 0.95))

model_starts = starts(model_ko, n=28*5)

model_ko_opt = optimise(model_ko, (x, y) -> weuclidean(x, y, data_ko.weight), data_ko.value, model_starts);

minimae = minimum.(model_ko_opt)
min_index = argmin(minimae)
#scatter(sort(minimae), lab="", mc=:black)
scatter(minimae, lab="", mc=:black)
scatter!([min_index], [minimum(model_ko_opt[min_index])], lab="", mc=:red)

best_ko_opt = model_ko_opt[argmin(minimum.(model_ko_opt))]

minimum(best_ko_opt)

model_ko_plot = dense(model_ko)
param_ko = parameter_dict(model_ko, transform(model_ko, best_ko_opt.minimizer))

plotlyjs()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylab="Fraction active", xlab="Time (days)")
for opt in filter(x -> minimum(x) < 7.5, model_ko_opt)
    sim = parameter_array(model_ko_plot, transform(model_ko_plot, opt.minimizer)) |> simulate(model_ko_plot) |> link(model_ko_plot)
    plot!(p1, model_ko_plot.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p2, model_ko_plot.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
end
#best_i = 2
#sim = parameter_array(model_ko_plot, transform(model_ko_plot, model_ko_opt[best_i].minimizer)) |> simulate(model_ko_plot) |> link(model_ko_plot)
#plot!(p1, model_ko_plot.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model_ko_plot.times, sim[:,2], lab="", lc=:green, lw=2)
sim = param_ko |> simulate(model_ko_plot) |> link(model_ko_plot)
plot!(p1, model_ko_plot.times, sim[:,1], lab="", lc=:red, lw=2)
plot!(p2, model_ko_plot.times, sim[:,2], lab="", lc=:red, lw=2)
@df @where(data_ko, :link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data_ko, :link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2)

ba = param_ko[:ba]
bb = param_ko[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, sim[:,1], b, lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
p

ba = param_ko[:ba]
bb = param_ko[:bb]
b_ko = 1 .- hill.(ba, bb, sim[:,1])
ba = param_wt[:ba]
bb = param_wt[:bb]
b_wt = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, sim[:,1], b_ko, lab="ko new", lw=2)
plot!(p, sim[:,1], b_wt, lab="wt new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
p

ba = param_ko[:ba]
bb = param_ko[:bb]
b_ko = 1 .- hill.(ba, bb, sim[:,1])
ba = param_wt[:ba]
bb = param_wt[:bb]
b_wt = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, model_ko_plot.times, b_ko, lab="ko new", lw=2)
plot!(p, model_wt_plot.times, b_wt, lab="wt new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :t, :b, label=first(group.genotype), lw=2)
end
p

param_ko

param_fix = Dict(
    :pₛ => 0.95,
    :ba => param_ko[:ba],
    :bb => param_ko[:bb],
)
model_wt_fixb = PopSelf((0.0, 700.0), Float64.(data_wt.age), data_wt.link, param_fix)

model_starts = starts(model_wt_fixb, n=28*5)

model_wt_fixb_opt = optimise(model_wt_fixb, (x, y) -> weuclidean(x, y, data_wt.weight), data_wt.value, model_starts);

best_wt_fixb_opt = model_wt_fixb_opt[argmin(minimum.(model_wt_fixb_opt))]

minimum(best_wt_fixb_opt)

model_wt_fixb_plot = dense(model_wt_fixb)
param_fixb_wt = parameter_dict(model_wt_fixb, transform(model_wt_fixb, best_wt_fixb_opt.minimizer))

plotlyjs()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylab="Fraction active", xlab="Time (days)")
for opt in model_wt_fixb_opt #filter(x -> minimum(x) < 7.5, model_wt_fixb_opt)
    sim = parameter_array(model_wt_fixb_plot, transform(model_wt_fixb_plot, opt.minimizer)) |> simulate(model_wt_fixb_plot) |> link(model_wt_fixb_plot)
    plot!(p1, model_wt_fixb_plot.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p2, model_wt_fixb_plot.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
end
best_i = 2
sim = parameter_array(model_wt_fixb_plot, transform(model_wt_fixb_plot, model_wt_opt[best_i].minimizer)) |> simulate(model_wt_fixb_plot) |> link(model_wt_fixb_plot)
plot!(p1, model_wt_fixb_plot.times, sim[:,1], lab="", lc=:green, lw=2)
plot!(p2, model_wt_fixb_plot.times, sim[:,2], lab="", lc=:green, lw=2)
sim = param_fixb_wt |> simulate(model_wt_fixb_plot) |> link(model_wt_fixb_plot)
plot!(p1, model_wt_fixb_plot.times, sim[:,1], lab="", lc=:red, lw=2)
plot!(p2, model_wt_fixb_plot.times, sim[:,2], lab="", lc=:red, lw=2)
@df @where(data_wt, :link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data_wt, :link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2)

ba = param_fixb_wt[:ba]
bb = param_fixb_wt[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, sim[:,1], b, lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
p

ba = param_fixb_wt[:ba]
bb = param_fixb_wt[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, model_wt_fixb_plot.times, b, lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :t, :b, label=first(group.genotype), lw=2)
end
p

param_wt

param_fix = Dict(
    :pₛ => 0.95,
    :ba => param_wt[:ba],
    :bb => param_wt[:bb],
)
model_ko_fixb = PopSelf((0.0, 700.0), Float64.(data_ko.age), data_ko.link, param_fix)

model_starts = starts(model_ko_fixb, n=28*5)

model_ko_fixb_opt = optimise(model_ko_fixb, (x, y) -> weuclidean(x, y, data_ko.weight), data_ko.value, model_starts);

best_ko_fixb_opt = model_ko_fixb_opt[argmin(minimum.(model_ko_fixb_opt))]

minimum(best_ko_fixb_opt)

model_ko_fixb_plot = dense(model_ko_fixb)
param_fixb_ko = parameter_dict(model_ko_fixb, transform(model_ko_fixb, best_ko_fixb_opt.minimizer))

plotlyjs()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylab="Fraction active", xlab="Time (days)")
for opt in model_ko_fixb_opt #filter(x -> minimum(x) < 7.5, model_ko_fixb_opt)
    sim = parameter_array(model_ko_fixb_plot, transform(model_ko_fixb_plot, opt.minimizer)) |> simulate(model_ko_fixb_plot) |> link(model_ko_fixb_plot)
    plot!(p1, model_ko_fixb_plot.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p2, model_ko_fixb_plot.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
end
#best_i = 2
#sim = parameter_array(model_ko_fixb_plot, transform(model_ko_fixb_plot, model_ko_opt[best_i].minimizer)) |> simulate(model_ko_fixb_plot) |> link(model_ko_fixb_plot)
#plot!(p1, model_ko_fixb_plot.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model_ko_fixb_plot.times, sim[:,2], lab="", lc=:green, lw=2)
sim = param_fixb_ko |> simulate(model_ko_fixb_plot) |> link(model_ko_fixb_plot)
plot!(p1, model_ko_fixb_plot.times, sim[:,1], lab="", lc=:red, lw=2)
plot!(p2, model_ko_fixb_plot.times, sim[:,2], lab="", lc=:red, lw=2)
@df @where(data_ko, :link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data_ko, :link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2)

ba = param_fixb_ko[:ba]
bb = param_fixb_ko[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, sim[:,1], b, lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
p

ba = param_fixb_ko[:ba]
bb = param_fixb_ko[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright)#, xscale=:log10)
plot!(p, model_ko_fixb_plot.times, b, lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :t, :b, label=first(group.genotype), lw=2)
end
p

param_fixb_ko

param = param_fixb_ko
times = model_ko_fixb_plot.times
model = model_ko_fixb_plot
pyplot()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylab="Fraction active", xlab="Time (days)")
p3 = plot(ylab="Activation rate", xlab="Time (days)")
p4 = plot(ylab="Self-renewal probability", xlab="Time (days)")
sim = param |> simulate(model) |> link(model)
plot!(p1, times, sim[:,1], lab="", lc=:red, lw=2)
plot!(p2, times, sim[:,2], lab="", lc=:red, lw=2)
plot!(p3, times, exponential_decay.(param[:r₀], param[:βr], times), lc=:red, lw=2, lab="")
plot!(p4, times, (1 .- hill.(param[:ba], param[:bb], sim[:,1])), lc=:red, lw=2, lab="")
@df @where(data_ko, :link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data_ko, :link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, p3, p4)

param

struct CombSelf <: ODEModel
    tspan::Tuple{Float64, Float64} 
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}} # pointer to thing to fix and value
    CombSelf(tspan, times, values,
                  fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end

function ratefun(model::CombSelf)
    function(du, u, p, t)
        _, wt_r₀, wt_βr, _, ko_r₀, ko_βr, ba, bb, pₛ = p
        Qw = u[1]; Aw = u[2]; Qk = u[3]; Ak = u[4]
        wt_r = exponential_decay(wt_r₀, wt_βr, t)
        ko_r = exponential_decay(ko_r₀, ko_βr, t)
        wt_b = 1-hill(ba, bb, Qw+Aw)
        ko_b = 1-hill(ba, bb, Qk+Ak)
        du[1] = dQw = -wt_r * Qw + 2 * wt_b * pₛ * Aw
        du[2] = dAw =  wt_r * Qw - pₛ * Aw
        du[3] = dQk = -ko_r * Qk + 2 * ko_b * pₛ * Ak
        du[4] = dAk =  ko_r * Qk - pₛ * Ak
    end
end

function initial(t::CombSelf, x::AbstractVector)
    wt_nsc₀, wt_r₀, _, ko_nsc₀, ko_r₀, _, ba, bb, pₛ = x
    wt_b₀ = 1-hill(ba, bb, wt_nsc₀)
    ko_b₀ = 1-hill(ba, bb, ko_nsc₀)
    wt_ratio = sqrt(((pₛ - wt_r₀)/(2*wt_r₀))^2 + (2*wt_b₀*pₛ) / wt_r₀)
    ko_ratio = sqrt(((pₛ - ko_r₀)/(2*ko_r₀))^2 + (2*ko_b₀*pₛ) / ko_r₀)
    vcat(wt_nsc₀ .* [1-1/(wt_ratio+1), 1/(wt_ratio+1)],  ko_nsc₀ .* [1-1/(ko_ratio+1), 1/(ko_ratio+1)] )
end

link(t::CombSelf, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]), x[:,3] .+ x[:,4], x[:,4] ./ (x[:,3] .+ x[:,4]))

parameter_names(t::Type{CombSelf}) = [:wt_nsc₀, :wt_r₀, :wt_βr, :ko_nsc₀, :ko_r₀, :ko_βr, :ba, :bb, :pₛ]

bounds(t::CombSelf) = [(100.0, 5000.0), (0.0, 1.0), (0.0, 0.1), (100.0, 5000.0), (0.0, 1.0), (0.0, 0.1), (10.0, 1000.0), (0.005, 2.0), (0.0, 1.0)]

output_names(t::CombSelf) = ["qNSC wt", "aNSC wt", "qNSC ko", "aNSC ko"]
link_names(t::CombSelf) = ["Total NSC wt", "Fraction active NSC wt", "Total NSC ko", "Fraction active NSC ko"]

data[!,"new_link"] = 2 .* Int64.(data.genotype .== "ifnko") .+ Int64.(data.name .== "active") .+ 1
data

model_co = CombSelf((0.0, 700.0), Float64.(data.age), data.new_link, Dict(:pₛ => 0.95))

model_starts = starts(model_co, n=28*5)

model_co_opt = optimise(model_co, (x, y) -> weuclidean(x, y, data.weight), data.value, model_starts);

minimum(minimum.(model_co_opt))^2 # sqeuclidean is the residual sum of squares. wsqeuclidean for some reason was really slow so I am redoing this with weuclidean^2

best_co_opt = model_co_opt[argmin(minimum.(model_co_opt))]
param_co = parameter_dict(model_co, transform(model_co, best_co_opt.minimizer))

?objective

param_co

free_parameters(model_wt)

objective(model_wt, (x, y) -> weuclidean(x, y, data_wt.weight), data_wt.value,  Dict(:nsc₀ => 5000.0, :r₀ => 0.672387, :βr => 0.0015538, :ba => 153.724, :bb => 0.0584954))^2

model = dense(model_co)
param = copy(param_co)
opts = model_co_opt
pyplot()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p3 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p4 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p5 = plot(ylab="Activation rate")
p6 = plot(ylab="Self-renewal probability")
#for opt in opts #filter(x -> minimum(x) < 7.5, model_ko_fixb_opt)
#    sim = parameter_array(model, transform(model, opt.minimizer)) |> simulate(model) |> link(model)
#    plot!(p1, model.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p2, model.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p3, model.times, sim[:,3], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p4, model.times, sim[:,4], lab="", lc=:gray, lw=1, alpha=0.4)
#end
#best_i = 28
#sim = parameter_array(model, transform(model, opts[best_i].minimizer)) |> simulate(model) |> link(model)
#plot!(p1, model.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model.times, sim[:,2], lab="", lc=:green, lw=2)
#plot!(p3, model.times, sim[:,3], lab="", lc=:green, lw=2)
#plot!(p4, model.times, sim[:,4], lab="", lc=:green, lw=2)
sim = param |> simulate(model) |> link(model)
plot!(p1, model.times, sim[:,1], lab="", lc=1, lw=2)
plot!(p2, model.times, sim[:,2], lab="", lc=1, lw=2)
plot!(p3, model.times, sim[:,3], lab="", lc=2, lw=2)
plot!(p4, model.times, sim[:,4], lab="", lc=2, lw=2)
plot!(p5, model.times, exponential_decay(param[:wt_r₀], param[:wt_βr]).(model.times), lab="WT")
plot!(p5, model.times, exponential_decay(param[:ko_r₀], param[:ko_βr]).(model.times), lab="KO")
plot!(p6, model.times, 1 .- hill(param[:ba], param[:bb]).(sim[:,1]), lab="", lc=:black)
@df @where(data, :new_link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 3) scatter!(p3, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 4) scatter!(p4, :age, :value, mc=:black, lab="")
plot(p1, p2, p3, p4, size=(600, 400))
#plot(p1, p2, p5, p3, p4, p6)
#savefig("b_shared.svg")
savefig("full_combination_b_fit.svg")

struct Comb2Self <: ODEModel
    tspan::Tuple{Float64, Float64} 
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}} # pointer to thing to fix and value
    Comb2Self(tspan, times, values,
                  fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end

function ratefun(model::Comb2Self)
    function(du, u, p, t)
        _, wt_r₀, wt_βr, _, ko_r₀, ko_βr, wt_ba, wt_bb, ko_ba, ko_bb, pₛ = p
        Qw = u[1]; Aw = u[2]; Qk = u[3]; Ak = u[4]
        wt_r = exponential_decay(wt_r₀, wt_βr, t)
        ko_r = exponential_decay(ko_r₀, ko_βr, t)
        wt_b = 1-hill(wt_ba, wt_bb, Qw+Aw)
        ko_b = 1-hill(ko_ba, ko_bb, Qk+Ak)
        du[1] = dQw = -wt_r * Qw + 2 * wt_b * pₛ * Aw
        du[2] = dAw =  wt_r * Qw - pₛ * Aw
        du[3] = dQk = -ko_r * Qk + 2 * ko_b * pₛ * Ak
        du[4] = dAk =  ko_r * Qk - pₛ * Ak
    end
end

function initial(t::Comb2Self, x::AbstractVector)
    wt_nsc₀, wt_r₀, _, ko_nsc₀, ko_r₀, _, wt_ba, wt_bb, ko_ba, ko_bb, pₛ = x
    wt_b₀ = 1-hill(wt_ba, wt_bb, wt_nsc₀)
    ko_b₀ = 1-hill(ko_ba, ko_bb, ko_nsc₀)
    wt_ratio = sqrt(((pₛ - wt_r₀)/(2*wt_r₀))^2 + (2*wt_b₀*pₛ) / wt_r₀)
    ko_ratio = sqrt(((pₛ - ko_r₀)/(2*ko_r₀))^2 + (2*ko_b₀*pₛ) / ko_r₀)
    vcat(wt_nsc₀ .* [1-1/(wt_ratio+1), 1/(wt_ratio+1)],  ko_nsc₀ .* [1-1/(ko_ratio+1), 1/(ko_ratio+1)] )
end

link(t::Comb2Self, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]), x[:,3] .+ x[:,4], x[:,4] ./ (x[:,3] .+ x[:,4]))

parameter_names(t::Type{Comb2Self}) = [:wt_nsc₀, :wt_r₀, :wt_βr, :ko_nsc₀, :ko_r₀, :ko_βr, :wt_ba, :wt_bb, :ko_ba, :ko_bb, :pₛ]

bounds(t::Comb2Self) = [(100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (10.0, 1000.0), (0.005, 2.0), (10.0, 1000.0), (0.005, 2.0), (0.0, 1.0)]

output_names(t::Comb2Self) = ["qNSC wt", "aNSC wt", "qNSC ko", "aNSC ko"]
link_names(t::Comb2Self) = ["Total NSC wt", "Fraction active NSC wt", "Total NSC ko", "Fraction active NSC ko"]

model_co2 = Comb2Self((0.0, 700.0), Float64.(data.age), data.new_link, Dict(:pₛ => 0.95))

model_starts = starts(model_co2, n=28*5)

model_co2_opt = optimise(model_co2, (x, y) -> weuclidean(x, y, data.weight), data.value, model_starts);

minimum(minimum.(model_co2_opt))^2

best_co2_opt = model_co2_opt[argmin(minimum.(model_co2_opt))]
param_co2 = parameter_dict(model_co2, transform(model_co2, best_co2_opt.minimizer))

model = dense(model_co2)
param = copy(param_co2)
opts = model_co2_opt
pyplot()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p3 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p4 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p5 = plot(ylab="Activation rate")
p6 = plot(ylab="Self-renewal probability")
#for opt in opts #filter(x -> minimum(x) < 7.5, model_ko_fixb_opt)
#    sim = parameter_array(model, transform(model, opt.minimizer)) |> simulate(model) |> link(model)
#    plot!(p1, model.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p2, model.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p3, model.times, sim[:,3], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p4, model.times, sim[:,4], lab="", lc=:gray, lw=1, alpha=0.4)
#end
#best_i = 28
#sim = parameter_array(model, transform(model, opts[best_i].minimizer)) |> simulate(model) |> link(model)
#plot!(p1, model.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model.times, sim[:,2], lab="", lc=:green, lw=2)
#plot!(p3, model.times, sim[:,3], lab="", lc=:green, lw=2)
#plot!(p4, model.times, sim[:,4], lab="", lc=:green, lw=2)
sim = param |> simulate(model) |> link(model)
plot!(p1, model.times, sim[:,1], lab="", lc=1, lw=2)
plot!(p2, model.times, sim[:,2], lab="", lc=1, lw=2)
plot!(p3, model.times, sim[:,3], lab="", lc=2, lw=2)
plot!(p4, model.times, sim[:,4], lab="", lc=2, lw=2)
plot!(p5, model.times, exponential_decay(param[:wt_r₀], param[:wt_βr]).(model.times), lab="WT")
plot!(p5, model.times, exponential_decay(param[:ko_r₀], param[:ko_βr]).(model.times), lab="KO")
plot!(p6, model.times, 1 .- hill(param[:wt_ba], param[:wt_bb]).(sim[:,1]), lab="", lc=1)
plot!(p6, model.times, 1 .- hill(param[:ko_ba], param[:ko_bb]).(sim[:,1]), lab="", lc=2)
@df @where(data, :new_link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 3) scatter!(p3, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 4) scatter!(p4, :age, :value, mc=:black, lab="")
plot(p1, p2, p3, p4, size=(600, 400))
#plot(p1, p2, p5, p3, p4, p6)
#savefig("b_shared.svg")
savefig("separate_b_fit.svg")

struct Comb3Self <: ODEModel
    tspan::Tuple{Float64, Float64} 
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}} # pointer to thing to fix and value
    Comb3Self(tspan, times, values,
                  fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end

function ratefun(model::Comb3Self)
    function(du, u, p, t)
        _, wt_r₀, wt_βr, _, ko_r₀, ko_βr, ba, wt_bb, ko_bb, pₛ = p
        Qw = u[1]; Aw = u[2]; Qk = u[3]; Ak = u[4]
        wt_r = exponential_decay(wt_r₀, wt_βr, t)
        ko_r = exponential_decay(ko_r₀, ko_βr, t)
        wt_b = 1-hill(ba, wt_bb, Qw+Aw)
        ko_b = 1-hill(ba, ko_bb, Qk+Ak)
        du[1] = dQw = -wt_r * Qw + 2 * wt_b * pₛ * Aw
        du[2] = dAw =  wt_r * Qw - pₛ * Aw
        du[3] = dQk = -ko_r * Qk + 2 * ko_b * pₛ * Ak
        du[4] = dAk =  ko_r * Qk - pₛ * Ak
    end
end

function initial(t::Comb3Self, x::AbstractVector)
    wt_nsc₀, wt_r₀, _, ko_nsc₀, ko_r₀, _, ba, wt_bb, ko_bb, pₛ = x
    wt_b₀ = 1-hill(ba, wt_bb, wt_nsc₀)
    ko_b₀ = 1-hill(ba, ko_bb, ko_nsc₀)
    wt_ratio = sqrt(((pₛ - wt_r₀)/(2*wt_r₀))^2 + (2*wt_b₀*pₛ) / wt_r₀)
    ko_ratio = sqrt(((pₛ - ko_r₀)/(2*ko_r₀))^2 + (2*ko_b₀*pₛ) / ko_r₀)
    vcat(wt_nsc₀ .* [1-1/(wt_ratio+1), 1/(wt_ratio+1)],  ko_nsc₀ .* [1-1/(ko_ratio+1), 1/(ko_ratio+1)] )
end

link(t::Comb3Self, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]), x[:,3] .+ x[:,4], x[:,4] ./ (x[:,3] .+ x[:,4]))

parameter_names(t::Type{Comb3Self}) = [:wt_nsc₀, :wt_r₀, :wt_βr, :ko_nsc₀, :ko_r₀, :ko_βr, :ba, :wt_bb, :ko_bb, :pₛ]

bounds(t::Comb3Self) = [(100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (10.0, 1000.0), (0.005, 2.0), (0.005, 2.0), (0.0, 1.0)]

output_names(t::Comb3Self) = ["qNSC wt", "aNSC wt", "qNSC ko", "aNSC ko"]
link_names(t::Comb3Self) = ["Total NSC wt", "Fraction active NSC wt", "Total NSC ko", "Fraction active NSC ko"]

model_co3 = Comb3Self((0.0, 700.0), Float64.(data.age), data.new_link, Dict(:pₛ => 0.95))

model_starts = starts(model_co3, n=28*5)

model_co3_opt = optimise(model_co3, (x, y) -> weuclidean(x, y, data.weight), data.value, model_starts);

minimum(minimum.(model_co3_opt))^2

best_co3_opt = model_co3_opt[argmin(minimum.(model_co3_opt))]
param_co3 = parameter_dict(model_co3, transform(model_co3, best_co3_opt.minimizer))

model = dense(model_co3)
param = copy(param_co3)
opts = model_co3_opt
pyplot()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p3 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p4 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p5 = plot(ylab="Activation rate")
p6 = plot(ylab="Self-renewal probability")
#for opt in opts #filter(x -> minimum(x) < 7.5, model_ko_fixb_opt)
#    sim = parameter_array(model, transform(model, opt.minimizer)) |> simulate(model) |> link(model)
#    plot!(p1, model.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p2, model.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p3, model.times, sim[:,3], lab="", lc=:gray, lw=1, alpha=0.4)
#    plot!(p4, model.times, sim[:,4], lab="", lc=:gray, lw=1, alpha=0.4)
#end
#best_i = 28
#sim = parameter_array(model, transform(model, opts[best_i].minimizer)) |> simulate(model) |> link(model)
#plot!(p1, model.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model.times, sim[:,2], lab="", lc=:green, lw=2)
#plot!(p3, model.times, sim[:,3], lab="", lc=:green, lw=2)
#plot!(p4, model.times, sim[:,4], lab="", lc=:green, lw=2)
sim = param |> simulate(model) |> link(model)
plot!(p1, model.times, sim[:,1], lab="", lc=1, lw=2)
plot!(p2, model.times, sim[:,2], lab="", lc=1, lw=2)
plot!(p3, model.times, sim[:,3], lab="", lc=2, lw=2)
plot!(p4, model.times, sim[:,4], lab="", lc=2, lw=2)
plot!(p5, model.times, exponential_decay(param[:wt_r₀], param[:wt_βr]).(model.times), lab="WT")
plot!(p5, model.times, exponential_decay(param[:ko_r₀], param[:ko_βr]).(model.times), lab="KO")
plot!(p6, model.times, 1 .- hill(param[:ba], param[:wt_bb]).(sim[:,1]), lab="", lc=1)
plot!(p6, model.times, 1 .- hill(param[:ba], param[:ko_bb]).(sim[:,1]), lab="", lc=2)
@df @where(data, :new_link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 3) scatter!(p3, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 4) scatter!(p4, :age, :value, mc=:black, lab="")
#plot(p1, p2, p3, p4)
plot(p1, p2, p3, p4, size=(600, 400))
#plot(p1, p2, p5, p3, p4, p6)
savefig("semi_shared_b.svg")



param_co3

p = plot(ylim=(0.455, 0.51), 
         xlim=(-100, 2600),
         xscale=:identity,
         xlab="Neural stem cells", ylab="Self renewal probability", 
         size=(300, 300))
Plots.abline!(0, 0.5, lab="", lc=:gray)
xs = 20:2500
#plot!(xs, x -> 1 .- hill(param_co3[:ba], param_co3[:wt_bb], x), lab="wt", lc=1)
#plot!(xs, x -> 1 .- hill(param_co3[:ba], param_co3[:ko_bb], x), lab="ko", lc=2)
plot!(xs, x -> 1 .- hill(param_co[:ba], param_co[:bb], x), lab="", lc=:black, lw=2)
#savefig("param-b.svg")
savefig("direct-b.svg")

p = plot(ylim=(0.455, 0.51), 
         xlim=(-100, 2600),
         xscale=:identity,
         xlab="Neural stem cells", ylab="Self renewal probability", 
         size=(300, 300))
Plots.abline!(0, 0.5, lab="", lc=:gray)
#xs = 20:2500
xs = vcat(collect(20:1000), collect(1000:2500))
ys = vcat(1 .- hill.(param_co3[:ba], param_co3[:ko_bb], 20:1000),
          1 .- hill.(param_co3[:ba], param_co3[:wt_bb], 1000:2500))
plot!(20:3500, x -> 1 .- hill(param_co3[:ba], param_co3[:ko_bb], x), lab="ko", lc=2, lw=2)
plot!(20:3500, x -> 1 .- hill(param_co3[:ba], param_co3[:wt_bb], x), lab="wt", lc=1, lw=2)
plot!(20:3500, x -> 1 .- hill(param_co[:ba], param_co[:bb], x), lab="IFN independent", lc=3, lw=2, style=:dot)
#plot!(xs, ys, lab="", lc=:black, lw=2)
plot!(20:1000, x -> 1 .- hill(param_co3[:ba], param_co3[:ko_bb], x), lab="IFN dependent", lc=:black, lw=2, alpha=0.7)
plot!(1000:2500, x -> 1 .- hill(param_co3[:ba], param_co3[:wt_bb], x), lab="", lc=:black, lw=2, alpha=0.7)
plot!([1000, 1000], [1 .- hill(param_co3[:ba], param_co3[:wt_bb], 1000), 1 .- hill(param_co3[:ba], param_co3[:ko_bb], 1000)], lab="", lc=:black, style=:dash, lw=2, alpha=0.7)
#plot!(xs, ys, lab="", lc=:black, lw=2)
#plot!(xs, x -> 1 .- hill(param_co[:ba], param_co[:bb], x), lab="", lc=:black, lw=2)
#savefig("param-b.svg")
savefig("complicated-b.svg")

param_co3



ba = param_fixb_wt[:ba]
bb = param_fixb_wt[:bb]
##nsc₀, r₀, _, ba, bb, pₛ = x
b = 1 .- hill.(ba, bb, sim[:,1])
p = plot(xflip=false, legend=:topright, ylab="b", xlab="NSC")#, xscale=:log10)
#plot!(p, sim[:,1], b, lab="new", lw=2)
plot!(p, sim[:,1], 1 .- hill(param_co[:ba], param_co[:bb]).(sim[:,1]), lab="new", lw=2)
for (i, group) in enumerate(groupby(sims, :genotype))
    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
end
p
#savefig("b-params.svg")

plot(param |> simulate(model), yscale=:log10, ylim=(10, 5_000))
#sim[:,4] |> plot

struct CombSelfPart <: ODEModel
    tspan::Tuple{Float64, Float64} 
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}} # pointer to thing to fix and value
    CombSelfPart(tspan, times, values,
                  fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end

function ratefun(model::CombSelfPart)
    function(du, u, p, t)
        _, wt_r₀, wt_βr, _, ko_r₀, ko_βr, ba, bb_wt, bb_ko, pₛ = p
        Qw = u[1]; Aw = u[2]; Qk = u[3]; Ak = u[4]
        wt_r = exponential_decay(wt_r₀, wt_βr, t)
        ko_r = exponential_decay(ko_r₀, ko_βr, t)
        wt_b = 1-hill(ba, bb_wt, Qw+Aw)
        ko_b = 1-hill(ba, bb_ko, Qk+Ak)
        du[1] = dQw = -wt_r * Qw + 2 * wt_b * pₛ * Aw
        du[2] = dAw =  wt_r * Qw - pₛ * Aw
        du[3] = dQk = -ko_r * Qk + 2 * ko_b * pₛ * Ak
        du[4] = dAk =  ko_r * Qk - pₛ * Ak
    end
end

function initial(t::CombSelfPart, x::AbstractVector)
    wt_nsc₀, wt_r₀, _, ko_nsc₀, ko_r₀, _, ba, bb_wt, bb_ko, pₛ = x
    wt_b₀ = 1-hill(ba, bb_wt, wt_nsc₀)
    ko_b₀ = 1-hill(ba, bb_ko, ko_nsc₀)
    wt_ratio = sqrt(((pₛ - wt_r₀)/(2*wt_r₀))^2 + (2*wt_b₀*pₛ) / wt_r₀)
    ko_ratio = sqrt(((pₛ - ko_r₀)/(2*ko_r₀))^2 + (2*ko_b₀*pₛ) / ko_r₀)
    vcat(wt_nsc₀ .* [1-1/(wt_ratio+1), 1/(wt_ratio+1)],  
         ko_nsc₀ .* [1-1/(ko_ratio+1), 1/(ko_ratio+1)])
end

link(t::CombSelfPart, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]), x[:,3] .+ x[:,4], x[:,4] ./ (x[:,3] .+ x[:,4]))

parameter_names(t::Type{CombSelfPart}) = [:wt_nsc₀, :wt_r₀, :wt_βr, :ko_nsc₀, :ko_r₀, :ko_βr, :ba, :bb_wt, :bb_ko, :pₛ]

bounds(t::CombSelfPart) = [(100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (100.0, 3000.0), (0.0, 1.0), (0.0, 0.1), (100.0, 1000.0), (0.01, 0.5), (0.01, 0.5), (0.0, 1.0)]

output_names(t::CombSelfPart) = ["qNSC wt", "aNSC wt", "qNSC ko", "aNSC ko"]
link_names(t::CombSelfPart) = ["Total NSC wt", "Fraction active NSC wt", "Total NSC ko", "Fraction active NSC ko"]

1-hill(150, 0.01, 10000)

#data[!,"new_link"] = 2 .* Int64.(data.genotype .== "ifnko") .+ Int64.(data.name .== "active") .+ 1
#data

existing_param = Dict(
    :wt_nsc₀ => 1986.0,
    :wt_r₀ => 0.8426,
    :wt_βr => 0.001891,
    :ko_nsc₀ => 2500.0,
    :ko_r₀ => 0.4928,
    :ko_βr => 1e-12,
    :pₛ => 0.95,
)

model_copart = CombSelfPart((0.0, 700.0), Float64.(data.age), data.new_link, existing_param)

model_starts = starts(model_copart, n=28*5)

model_copart_opt = optimise(model_copart, (x, y) -> weuclidean(x, y, data.weight), data.value, model_starts);

best_copart_opt = model_copart_opt[argmin(minimum.(model_copart_opt))]
param_copart = parameter_dict(model_copart, transform(model_copart, best_copart_opt.minimizer))

free_parameters(model_copart)

model = dense(model_copart)
param = copy(param_copart)
opts = model_copart_opt
pyplot()
p1 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p2 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p3 = plot(yscale=:log10, ylab="Total NSC", xlab="Time (days)")
p4 = plot(ylim=(0, 0.8), ylab="Fraction active", xlab="Time (days)")
p5 = plot(ylab="Activation rate")
p6 = plot(ylab="Self-renewal probability")
for opt in opts #filter(x -> minimum(x) < 7.5, model_ko_fixb_opt)
    sim = parameter_array(model, transform(model, opt.minimizer)) |> simulate(model) |> link(model)
    plot!(p1, model.times, sim[:,1], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p2, model.times, sim[:,2], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p3, model.times, sim[:,3], lab="", lc=:gray, lw=1, alpha=0.4)
    plot!(p4, model.times, sim[:,4], lab="", lc=:gray, lw=1, alpha=0.4)
end
#best_i = 28
#sim = parameter_array(model, transform(model, opts[best_i].minimizer)) |> simulate(model) |> link(model)
#plot!(p1, model.times, sim[:,1], lab="", lc=:green, lw=2)
#plot!(p2, model.times, sim[:,2], lab="", lc=:green, lw=2)
#plot!(p3, model.times, sim[:,3], lab="", lc=:green, lw=2)
#plot!(p4, model.times, sim[:,4], lab="", lc=:green, lw=2)
sim = param |> simulate(model) |> link(model)
plot!(p1, model.times, sim[:,1], lab="", lc=:red, lw=2)
plot!(p2, model.times, sim[:,2], lab="", lc=:red, lw=2)
plot!(p3, model.times, sim[:,3], lab="", lc=:red, lw=2)
plot!(p4, model.times, sim[:,4], lab="", lc=:red, lw=2)
plot!(p5, model.times, exponential_decay(param[:wt_r₀], param[:wt_βr]).(model.times), lab="WT")
plot!(p5, model.times, exponential_decay(param[:ko_r₀], param[:ko_βr]).(model.times), lab="KO")
plot!(p6, model.times, 1 .- hill(param[:ba], param[:bb_wt]).(sim[:,1]), lab="WT")
plot!(p6, model.times, 1 .- hill(param[:ba], param[:bb_ko]).(sim[:,1]), lab="KO")
@df @where(data, :new_link .== 1) scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 2) scatter!(p2, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 3) scatter!(p3, :age, :value, mc=:black, lab="")
@df @where(data, :new_link .== 4) scatter!(p4, :age, :value, mc=:black, lab="")
plot(p1, p2, p5, p3, p4, p6)

plot(xscale=:identity)
plot!(sim[:,1], 1 .- hill(param[:ba], param[:bb_wt]).(sim[:,1]), lab="WT")
plot!(sim[:,1], 1 .- hill(param[:ba], param[:bb_ko]).(sim[:,1]), lab="KO")
plot!(sim[:,1], 1 .- hill(param_co[:ba], param_co[:bb]).(sim[:,1]), lab="Shared")
#plot!(p6, model.times, 1 .- hill(param[:ba], param[:bb_ko]).(sim[:,1]), lab="KO")ba = param_fixb_wt[:ba]
#bb = param_fixb_wt[:bb]
###nsc₀, r₀, _, ba, bb, pₛ = x
#b = 1 .- hill.(ba, bb, sim[:,1])
#p = plot(xflip=false, legend=:topright, ylab="b", xlab="NSC")#, xscale=:log10)
##plot!(p, sim[:,1], b, lab="new", lw=2)
#plot!(p, sim[:,1], 1 .- hill(param[:ba], param[:bb]).(sim[:,1]), lab="new", lw=2)
#for (i, group) in enumerate(groupby(sims, :genotype))
#    @df group plot!(p, :counts, :b, label=first(group.genotype), lw=2)
#end
#p
#s

DataFrame(
    nsc=sim[:,1], 
    b_wt=hill(param[:ba], param[:bb_wt]).(sim[:,1]),
    b_ko=hill(param[:ba], param[:bb_ko]).(sim[:,1]),
    b_shared=hill(param_co[:ba], param_co[:bb]).(sim[:,1]),
)

sim[:,3]

minimae = minimum.(model_copart_opt)
scatter(minimae, mc=:black, lab="")

model_wt |> free_parameters



bmodel_wt = PopSelf((0.0, 700.0), Float64.(data_wt.age), data_wt.link,  Dict(x => param_wt[x] for x in setdiff(keys(param_wt), [:ba, :bb])))

free_parameters(bmodel_wt)

simulate(bmodel_wt, Dict(:ba => 150.0, :bb => 0.1))

free_parameters(bmodel_wt)

parameter_array(bmodel_wt, [1, 2])

@model function wt_bayes(data, bmodel)
    ba ~ Poisson(150)
    bb ~ Uniform(0.01, 0.1) #Normal() # 0.01 0.1
    p = parameter_array(bmodel, [ba, bb], eltype=eltype(bb))
    sim = simulate(bmodel, p) |> link(bmodel)
    for i in 1:length(data)
        point = bmodel.values[i]
        λ = Int64(round(sim[i,1]))
        if point == 1
            data[i] ~ Poisson(λ)
        else
            data[i] ~ Binomial(λ, sim[i,2])
        end
    end
end
#chain = sample(wt_bayes(data_wt.value), HMC(0.05, 10), 1000)

model = wt_bayes(data_wt.value, bmodel_wt)

wt_bayes_mle = optimize(model, MLE())

wt_bayes_map = optimize(model, MAP())

prob"ba = 146.0, bb = 0.0742176 | model=model, data=data_wt.value, bmodel=bmodel_wt"

Turing.setadbackend(:forwarddiff)

chain = sample(model, HMC(0.05, 10), 1000, init_theta=wt_bayes_mle.values.array);

chain

histogram(chain)





simulate(bmodel_wt, param_wt) |> link(bmodel_wt)

bmodel_wt.values


