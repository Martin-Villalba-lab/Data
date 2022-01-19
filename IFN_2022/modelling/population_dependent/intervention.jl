using Revise

using DifferentialEquations
using Optim
using Statistics
using Distances

using CSV
using DataFrames
using DataFramesMeta
using Chain: @chain
using Pipe: @pipe
using DataStructures

using Plots
using StatsPlots
pyplot()

includet("models/base.jl")

data = map(["wt", "ifnko"]) do geno
    active = @linq CSV.read("data_old/data_$(geno)_active.csv", DataFrame) |>
        transform(name="active") |>
        select(genotype=:genotype, age=:age, name=:name, value=:active)
    counts = @linq CSV.read("data_old/data_$(geno)_counts.csv", DataFrame) |>
        transform(name="total") |>
        select(genotype=:genotype, age=:age, name=:name, value=:count)
    vcat([active, counts]...)
end
data = vcat(data...)

sim_ifnko_both = CSV.read("data_old/solution_ifnko_self_renewal_and_quiescence.csv", DataFrame)
sim_wt_both = CSV.read("data_old/solution_wt_self_renewal_and_quiescence.csv", DataFrame)
sim = vcat(@transform(sim_wt_both, genotype="wt"), @transform(sim_ifnko_both, genotype="ko"))

p = plot(yscale=:log10, ylab="rate of TAP creation by NSC division", xlab="Time (days)")
for group in groupby(sim, :genotype)
    group = @transform(group, tapo=(1 .- :b) .* :counts .* :active .* 2 * 0.95)
    @df group plot!(p, :t, :tapo, label=unique(group.genotype)[1], lw=2)
end
p

@chain sim begin
    @transform tapo=(1 .- :b) .* :counts .* :active .* 2 * 0.95 
    groupby(:genotype)
    @combine mean=mean(:tapo)
end

includet("models/nsc_kalamakis.jl")

times = collect(1:0.1:700.0)
model = KalamakisFull((0.0, 700.0), times, zeros(Int64, length(times)), Dict(:pₛ => 0.95))

parameter_names(model)

param = Dict(
    :wt => Dict(
        :nsc₀ => 1986.0,
        :r₀ => 0.8426,
        :βr => 0.001891,
        :b₀ => 0.4822,
        :βb => 0.004862
    ),
    :ko => Dict(
        :nsc₀ => 2500.0,
        :r₀ => 0.4928,
        :βr => 1e-12,
        :b₀ => 0.4455,
        :βb => 0.01288
    )
)

sim = simulate(model, param[:wt]) |> link(model)
ylabs = link_names(model)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, title="wt", yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, xlab="Time (days)")

sim = simulate(model, param[:ko]) |> link(model)
ylabs = link_names(model)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, title="ko", yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "ifnko", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "ifnko", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, xlab="Time (days)")

param_rt = Dict(
    geno => exponential_decay(params[:r₀], params[:βr])
    for (geno, params) in param
)
param_bt = Dict(
    geno => flattening_curve(params[:b₀], params[:βb])
    for (geno, params) in param
)

p1 = plot(times, param_rt[:wt], lab="wt", ylab="Activation rate")
plot!(p1, times, param_rt[:ko], lab="ko")
p2 = plot(times, param_bt[:wt], lab="wt", ylab="Self-renewal probability", legend=:bottomright)
plot!(p2, times, param_bt[:ko], lab="ko")
plot(p1, p2, xlab="Time (days)", lw=2)

hill(ka, n, x) = 1/(1+(ka/x)^n)
hill(ka, n) = x -> hill(ka, n, x)

metrics_neurogen = DataFrame(itime = collect(0:1:660), type="neurogenesis");
metrics_loss = DataFrame(itime = collect(0:1:660), type="aged loss");

struct ChangedBetaSelf <: ODEModel
    tspan::Tuple{Float64, Float64}
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}}
    ChangedBetaSelf(tspan, times, values, fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end
function ratefun(model::ChangedBetaSelf)
    function(du, u, p, t)
        _, _, βr, kab_wt, kab_ko, nb_wt, nb_ko, pₛ, t₁ = p
        Q = u[1]; A = u[2]; r = u[3]
        b = 1 - hill(t < t₁ ? kab_wt : kab_ko, 
                     t < t₁ ? nb_wt : nb_ko, 
                     Q+A)
        du[1] = dQ = -r * Q + 2* b * pₛ * A
        du[2] = dA = r * Q - pₛ * A
        if t < t₁
            du[3] = dr = -βr * r
        else
            du[3] = dr = 0
        end
    end
end
function initial(t::ChangedBetaSelf, x::AbstractVector)
    nsc₀, r₀, _, kab_wt, _, nb_wt, _, pₛ, _ = x
    b₀ = hill(kab_wt, nb_wt, 0)
    ratio = sqrt(((pₛ - r₀)/(2*r₀))^2 + (2*b₀*pₛ) / r₀)
    vcat(nsc₀ .* [1-1/(ratio+1), 1/(ratio+1)], [r₀])
end
link(t::ChangedBetaSelf, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]))
parameter_names(t::Type{ChangedBetaSelf}) = [:nsc₀, :r₀, :βr, :kab_wt, :kab_ko, :nb_wt, :nb_ko, :pₛ, :t₁]
bounds(t::ChangedBetaSelf) = [(100.0, 10000.0), (0.0, 1.0), (0.0, 0.1), (0.0, 1000.0), (0.0, 1000.0), (0.01, 0.1), (0.01, 0.1), (0.0, 1.0), (0.0, 700.0)]
output_names(t::ChangedBetaSelf) = ["qNSC", "aNSC", "r"]
link_names(t::ChangedBetaSelf) = ["Total NSC", "Fraction active NSC"]

param_beta = Dict( # from partial cofit
    #:nsc₀ => 1986.0,
    :nsc₀ => 3000.0,
    #:r₀ => 0.8426,
    :r₀ => 0.806991,
    #:βr => 0.001891,
    :βr => 0.00184341,
    :kab_wt => 155.445,
    :kab_ko => 148.364,
    :nb_wt => 0.0399102,
    :nb_ko => 0.0843197,
    :pₛ => 0.95,
)

times = collect(0:0.1:700.0)
betamodel = ChangedBetaSelf((0.0, 700.0), times, zeros(Int64, length(times)), param_beta)

free_parameters(betamodel)



t₁ = 30.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

t₁ = 400.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)
#savefig("example_3.svg")
savefig("betamodel_interaction.svg")

t₁ = 600.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

#pgfplots()
pyplot()
itimes = collect(0:1:660)
rawsim = simulate(betamodel, Dict(:t₁ => 800.0)) # No intervention model
nonsim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab="Relative NSCs", xlab="Age (days)")
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
p6 = plot(ylab="Relative rate", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen_wt
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(betamodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
    plot!(p6, times[changed], neurogen[changed] ./ neurogen_wt[changed], lab="", lc=colours[i], la=0.9)
    tot_rel[!,"d$(Int64(round(itime)))"] = sim[:,1] ./ tot_wt
    neurogen_rel[!,"d$(Int64(round(itime)))"] = neurogen ./ neurogen_wt
    tot_abs[!,"d$(Int64(round(itime)))"] = sim[:,1]
    neurogen_abs[!,"d$(Int64(round(itime)))"] = neurogen
end
# Plot previous simulation
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,1] ./ nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p3, times, rawsim[:,3], lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen_wt, lc=:black, lw=2, lab="")
plot!(p6, times, neurogen_wt ./ neurogen_wt, lc=:black, lw=2, lab="")
# Plot data
#@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
#@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p7 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
steps = round(minimum(total_neurogen); sigdigits=1):0.05:round(maximum(total_neurogen); sigdigits=1)
p8 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(100*x))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5 p6;
    p7 p8;
]
plot(p1, p2, p4, p3, p5, p6, p7, p8, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("freeze_activation_individual_self.svg")
tot_rel |> CSV.write("sims/freeze_activation_individual_self_relative_nsc.csv")
neurogen_rel |> CSV.write("sims/freeze_activation_individual_self_relative_neurogen.csv")
tot_abs |> CSV.write("sims/freeze_activation_individual_self_absolute_nsc.csv")
neurogen_abs |> CSV.write("sims/freeze_activation_individual_self_absolute_neurogen.csv")

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
yticks = 50:5:115
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=(collect(yticks), ["$x\\%" for x in yticks]))
metrics_neurogen[!,"beta_fullself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)

itimes = collect(0:10:699)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
p1 = plot(times, sim[:,1], lab="", lc=:gray, lw=2, yscale=:log10, ylab=ylabs[1])
p2 = plot(times, sim[:,2], lab="", lc=:gray, lw=2, ylab=ylabs[2])
p3 = plot(times, param_rt[:wt], lab="", ylab="Activation rate", lc=:gray, lw=2)
#p4 = plot(times, param_bt[:wt], lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
p4 = plot(times, hillfun(800).(sim[:,1], times), lab="", 
          ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
for (i, t₁) in enumerate(itimes)
    step = times[2] - times[1]
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    sim = link(betamodel, rawsim)
    has_changed = times .> t₁
    has_changed[max(1, findfirst(has_changed)-2):end] .= true
    plot!(p1, times[has_changed], sim[has_changed,1], lab="", lw=1, lc=colours[i]) #t₁=$t₁
    plot!(p2, times[has_changed], sim[has_changed,2], lab="", lw=1, lc=colours[i])
    plot!(p3, times[has_changed], rawsim[has_changed,3], lab="", lw=1, lc=colours[i])
    plot!(p4, times[has_changed], hillfun(t₁).(sim[:,1], times)[has_changed], lab="", lw=1, lc=colours[i])
end
#plot!(p1, times, sim[:,1], lab="", lc=:red, lw=2)
@df @where(data, :genotype .== "wt", :name .== "total")  scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, p4, p3, xlab="Time (days)", size=(600, 600))

#itimes = collect(0:1:660)#[100, 200]#, 60, 100, 300, 600]
itimes = metrics_loss.itime
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
regsim = link(betamodel, rawsim)
nsc660 = regsim[times .== 660.0, 1][1]
nscs = map(itimes) do t₁
    rawsim = simulate(betamodel, Dict(:t₁ => t₁))
    regsim = link(betamodel, rawsim)
    nsc660 .- regsim[times .== 660.0, 1][1]
end
metrics_loss[!,"beta_fullself"] = nscs
plot(itimes, nscs, lc=:black, lab="", xlab="Intervention Time (days)", ylab="Stem cells lost", title="Loss of stem cells from intervention")

metrics_loss

itimes = collect(0:10:600)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
p = plot(xlab="Time (days)", ylab="Rate of TAP production from NSCs", yscale=:log10)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #has_changed = times .> t₁
    plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)), lab="", lc=colours[i])
end
rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)), lab="WT", lc=:gray, lw=2)
#plot!(p, sim_ifnko_both.t, sim_ifnko_both.active .* sim_ifnko_both.counts .* sim_ifnko_both.b .* 2 .* 0.95, lab="KO", lc=:red, lw=2)
p
#savefig("plots/scenario1-lifelong-neurogenesis.svg")

itimes = collect(0:10:600)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
p = plot(xlab="Time (days)", ylab="Relative progenitor production", title="Progenitor production compared to wildtype")#, yscale=:log10)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
neurogen = rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times))
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #has_changed = times .> t₁
    plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)) ./ neurogen, lab="", lc=colours[i], cbar=true)
end
rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)) ./ neurogen, lab="WT", lc=:black, lw=2,
      yticks=(0.5:0.25:2.5, ["$(Int64(round(x*100)))\\%" for x in 0.5:0.25:2.5]),
      xticks=(0:100:700)
)
#plot!(p, sim_ifnko_both.t, sim_ifnko_both.active .* sim_ifnko_both.counts .* sim_ifnko_both.b .* 2 .* 0.95, lab="KO", lc=:red, lw=2)
p
#savefig("plots/scenario1-lifelong-neurogenesis.svg")

itimes = collect(0:10:699)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
tot = sim[:,1]
ylabs = link_names(betamodel)
p1 = plot(lab="", lc=:gray, lw=2, ylab=ylabs[1], xlab="Time (days)", title="Total stem cells compared to wildtype")
for (i, t₁) in enumerate(itimes)
    step = times[2] - times[1]
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    sim = link(betamodel, rawsim)
    plot!(p1, times, sim[:,1] ./ tot, lab="", lw=1, lc=colours[i]) #t₁=$t₁
end
plot!(p1, times, tot ./ tot, lab="", lc=:black, lw=2,
    yticks=(0.5:0.1:1.0, ["$(Int64(x*100))\\%" for x in 0.5:0.1:1.0])
)
#@df @where(data, :genotype .== "wt", :name .== "total")  scatter!(p1, :age, :value, mc=:black, lab="")
p1

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
yticks = 50:5:115
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=(collect(yticks), ["$x\\%" for x in yticks]))
metrics_neurogen[!,"beta_fullself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)

param_beta = Dict( # from cofit
    #:nsc₀ => 1986.0,
    :nsc₀ => 3000.0,
    #:r₀ => 0.8426,
    :r₀ => 0.839538,
    #:βr => 0.001891,
    :βr => 0.00188605,
    :kab_wt => 153.81,
    :kab_ko => 153.81,
    :nb_wt => 0.039907,
    :nb_ko => 0.0841552,
    :pₛ => 0.95,
)



times = collect(0:1:700.0)
betamodel = ChangedBetaSelf((0.0, 700.0), times, zeros(Int64, length(times)), param_beta)

free_parameters(betamodel)



t₁ = 30.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

t₁ = 400.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)
savefig("betamodel_static.svg")

t₁ = 600.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

itimes = collect(0:10:699)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
p1 = plot(times, sim[:,1], lab="", lc=:gray, lw=2, yscale=:log10, ylab=ylabs[1])
p2 = plot(times, sim[:,2], lab="", lc=:gray, lw=2, ylab=ylabs[2])
p3 = plot(times, param_rt[:wt], lab="", ylab="Activation rate", lc=:gray, lw=2)
#p4 = plot(times, param_bt[:wt], lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
p4 = plot(times, hillfun(800).(sim[:,1], times), lab="", 
          ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    sim = link(betamodel, rawsim)
    has_changed = times .> t₁
    has_changed[max(1, findfirst(has_changed)-2):end] .= true
    plot!(p1, times[has_changed], sim[has_changed,1], lab="", lw=1, lc=colours[i]) #t₁=$t₁
    plot!(p2, times[has_changed], sim[has_changed,2], lab="", lw=1, lc=colours[i])
    plot!(p3, times[has_changed], rawsim[has_changed,3], lab="", lw=1, lc=colours[i])
    plot!(p4, times[has_changed], hillfun(t₁).(sim[:,1], times)[has_changed], lab="", lw=1, lc=colours[i])
end
#plot!(p1, times, sim[:,1], lab="", lc=:red, lw=2)
@df @where(data, :genotype .== "wt", :name .== "total")  scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, p4, p3, xlab="Time (days)", size=(600, 600))

#pgfplots()
pyplot()
itimes = collect(0:1:660)
rawsim = simulate(betamodel, Dict(:t₁ => 800.0)) # No intervention model
nonsim = link(betamodel, rawsim)
sim = nonsim
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab="Relative NSCs", xlab="Age (days)")
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
p6 = plot(ylab="Relative rate", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen_wt
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(betamodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
    plot!(p6, times[changed], neurogen[changed] ./ neurogen_wt[changed], lab="", lc=colours[i], la=0.9)
    tot_rel[!,"d$(Int64(round(itime)))"] = sim[:,1] ./ tot_wt
    neurogen_rel[!,"d$(Int64(round(itime)))"] = neurogen ./ neurogen_wt
    tot_abs[!,"d$(Int64(round(itime)))"] = sim[:,1]
    neurogen_abs[!,"d$(Int64(round(itime)))"] = neurogen
end
# Plot previous simulation
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,1] ./ nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p3, times, rawsim[:,3], lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen_wt, lc=:black, lw=2, lab="")
plot!(p6, times, neurogen_wt ./ neurogen_wt, lc=:black, lw=2, lab="")
# Plot data
#@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
#@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p7 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
steps = round(minimum(total_neurogen); sigdigits=1):0.05:round(maximum(total_neurogen); sigdigits=1)
p8 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(100*x))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5 p6;
    p7 p8;
]
plot(p1, p2, p4, p3, p5, p6, p7, p8, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_semishared_self.tex")
savefig("freeze_activation_semishared_self.svg")
tot_rel |> CSV.write("sims/freeze_activation_semishared_self_relative_nsc.csv")
neurogen_rel |> CSV.write("sims/freeze_activation_semishared_self_relative_neurogen.csv")
tot_abs |> CSV.write("sims/freeze_activation_semishared_self_absolute_nsc.csv")
neurogen_abs |> CSV.write("sims/freeze_activation_semishared_self_absolute_neurogen.csv")

#pgfplots()
pyplot()
rawsim = simulate(betamodel, Dict(:t₁ => 800.0)) # No intervention model
itimes = collect(0:10:660)
nonsim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
# ???
neurogen = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(betamodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    plot!(p2, times[changed], sim[changed,2], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
end
# Plot previous sims
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,2], lab="", lc=:black, lw=2)
plot!(p3, times, rawsim[:,3], lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen, lc=:black, lw=2, lab="")
# Plot data
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p6 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
neurogen_range = maximum(total_neurogen) - minimum(total_neurogen)
steps = 0.95:0.05:1.2
p7 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(round(100*x)))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5{.5h} [p6;
             p7]
]
plot(p1, p2, p4, p3, p5, p6, p7, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("freeze_activation_semishared_self.svg")

itimes = collect(0:1:660)#[100, 200]#, 60, 100, 300, 600]
itimes = metrics_loss.itime
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
regsim = link(betamodel, rawsim)
nsc660 = regsim[times .== 660.0, 1][1]
nscs = map(itimes) do t₁
    rawsim = simulate(betamodel, Dict(:t₁ => t₁))
    regsim = link(betamodel, rawsim)
    nsc660 .- regsim[times .== 660.0, 1][1]
end
metrics_loss[!,"beta_partself"] = nscs
plot(itimes, nscs, lc=:black, lab="", xlab="Intervention Time (days)", ylab="Stem cells lost", title="Loss of stem cells from intervention")

itimes = collect(0:10:600)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
p = plot(xlab="Time (days)", ylab="Rate of TAP production from NSCs", yscale=:log10)
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #has_changed = times .> t₁
    plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)), lab="", lc=colours[i])
end
rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)), lab="WT", lc=:gray, lw=2)
#plot!(p, sim_ifnko_both.t, sim_ifnko_both.active .* sim_ifnko_both.counts .* sim_ifnko_both.b .* 2 .* 0.95, lab="KO", lc=:red, lw=2)
p
#savefig("plots/scenario1-lifelong-neurogenesis.svg")

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
yticks = 50:5:115
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=(collect(yticks), ["$x\\%" for x in yticks]))
metrics_neurogen[!,"beta_partself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)

param_beta = Dict( # from cofit
    #:nsc₀ => 1986.0,
    :nsc₀ => 5000.0,
    #:r₀ => 0.8426,
    :r₀ => 0.672387,
    #:βr => 0.001891,
    :βr => 0.0015538,
    :kab_wt => 153.724,
    :kab_ko => 153.724,
    :nb_wt => 0.0584954,
    :nb_ko => 0.0584954,
    #:nb_ko => 0.0765863,
    :pₛ => 0.95,
)



times = collect(0:1:700.0)
betamodel = ChangedBetaSelf((0.0, 700.0), times, zeros(Int64, length(times)), param_beta)

free_parameters(betamodel)



t₁ = 30.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

t₁ = 400.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)
savefig("betamodel_static.svg")

t₁ = 600.0
rawsim = simulate(betamodel, Dict(:t₁ => t₁))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, rawsim[:,3], lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

itimes = collect(0:10:699)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
p1 = plot(times, sim[:,1], lab="", lc=:gray, lw=2, yscale=:log10, ylab=ylabs[1])
p2 = plot(times, sim[:,2], lab="", lc=:gray, lw=2, ylab=ylabs[2])
p3 = plot(times, param_rt[:wt], lab="", ylab="Activation rate", lc=:gray, lw=2)
#p4 = plot(times, param_bt[:wt], lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
p4 = plot(times, hillfun(800).(sim[:,1], times), lab="", 
          ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    sim = link(betamodel, rawsim)
    has_changed = times .> t₁
    has_changed[max(1, findfirst(has_changed)-2):end] .= true
    plot!(p1, times[has_changed], sim[has_changed,1], lab="", lw=1, lc=colours[i]) #t₁=$t₁
    plot!(p2, times[has_changed], sim[has_changed,2], lab="", lw=1, lc=colours[i])
    plot!(p3, times[has_changed], rawsim[has_changed,3], lab="", lw=1, lc=colours[i])
    plot!(p4, times[has_changed], hillfun(t₁).(sim[:,1], times)[has_changed], lab="", lw=1, lc=colours[i])
end
#plot!(p1, times, sim[:,1], lab="", lc=:red, lw=2)
@df @where(data, :genotype .== "wt", :name .== "total")  scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, p4, p3, xlab="Time (days)", size=(600, 600))

#pgfplots()
pyplot()
itimes = collect(0:1:660)
rawsim = simulate(betamodel, Dict(:t₁ => 800.0)) # No intervention model
nonsim = link(betamodel, rawsim)
sim = nonsim
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab="Relative NSCs", xlab="Age (days)")
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
p6 = plot(ylab="Relative rate", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen_wt
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(betamodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
    plot!(p6, times[changed], neurogen[changed] ./ neurogen_wt[changed], lab="", lc=colours[i], la=0.9)
    tot_rel[!,"d$(Int64(round(itime)))"] = sim[:,1] ./ tot_wt
    neurogen_rel[!,"d$(Int64(round(itime)))"] = neurogen ./ neurogen_wt
    tot_abs[!,"d$(Int64(round(itime)))"] = sim[:,1]
    neurogen_abs[!,"d$(Int64(round(itime)))"] = neurogen
end
# Plot previous simulation
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,1] ./ nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p3, times, rawsim[:,3], lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen_wt, lc=:black, lw=2, lab="")
plot!(p6, times, neurogen_wt ./ neurogen_wt, lc=:black, lw=2, lab="")
# Plot data
#@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
#@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p7 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
steps = round(minimum(total_neurogen); sigdigits=1):0.05:round(maximum(total_neurogen); sigdigits=1)
p8 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(100*x))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5 p6;
    p7 p8;
]
plot(p1, p2, p4, p3, p5, p6, p7, p8, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_shared_self.tex")
savefig("freeze_activation_shared_self.svg")
tot_rel |> CSV.write("sims/freeze_activation_shared_self_relative_nsc.csv")
neurogen_rel |> CSV.write("sims/freeze_activation_shared_self_relative_neurogen.csv")
tot_abs |> CSV.write("sims/freeze_activation_shared_self_absolute_nsc.csv")
neurogen_abs |> CSV.write("sims/freeze_activation_shared_self_absolute_neurogen.csv")

#pgfplots()
pyplot()
rawsim = simulate(betamodel, Dict(:t₁ => 800.0)) # No intervention model
itimes = collect(0:10:660)
nonsim = link(betamodel, rawsim)
ylabs = link_names(betamodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
#p2 = plot(ylab=ylabs[2], xlab="Age (days)")
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(betamodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times[changed], sim[changed,2], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
end
# Plot previous sims
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,2], lab="", lc=:black, lw=2)
plot!(p3, times, rawsim[:,3], lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen, lc=:black, lw=2, lab="")
# Plot data
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p6 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
neurogen_range = maximum(total_neurogen) - minimum(total_neurogen)
steps = 0.95:0.05:1.2
p7 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(round(100*x)))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5{.5h} [p6;
             p7]
]
plot(p1, p2, p4, p3, p5, p6, p7, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("freeze_activation_shared_self.svg")

itimes = collect(0:1:660)#[100, 200]#, 60, 100, 300, 600]
itimes = metrics_loss.itime
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
regsim = link(betamodel, rawsim)
nsc660 = regsim[times .== 660.0, 1][1]
nscs = map(itimes) do t₁
    rawsim = simulate(betamodel, Dict(:t₁ => t₁))
    regsim = link(betamodel, rawsim)
    nsc660 .- regsim[times .== 660.0, 1][1]
end
metrics_loss[!,"beta_noself"] = nscs
plot(itimes, nscs, lc=:black, lab="", xlab="Intervention Time (days)", ylab="Stem cells lost", title="Loss of stem cells from intervention")

itimes = collect(0:10:600)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(betamodel, Dict(:t₁ => 700.0))
sim = link(betamodel, rawsim)
p = plot(xlab="Time (days)", ylab="Rate of TAP production from NSCs", yscale=:log10)
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #has_changed = times .> t₁
    plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)), lab="", lc=colours[i])
end
rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)), lab="WT", lc=:gray, lw=2)
#plot!(p, sim_ifnko_both.t, sim_ifnko_both.active .* sim_ifnko_both.counts .* sim_ifnko_both.b .* 2 .* 0.95, lab="KO", lc=:red, lw=2)
p
#savefig("plots/scenario1-lifelong-neurogenesis.svg")

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(betamodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(betamodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
yticks = 50:5:115
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=(collect(yticks), ["$x\\%" for x in yticks]))
metrics_neurogen[!,"beta_noself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)



function broken_exponential_decay(x₀, β, x₁, t₁, t)
    if t < t₁
        x₀ * exp(-β*t)
    else
        x₁
    end
end
broken_exponential_decay(x₀, β, x₁, t₁) = t -> broken_exponential_decay(x₀, β, x₁, t₁, t)

struct ChangedValueSelf <: ODEModel
    tspan::Tuple{Float64, Float64}
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}}
    ChangedValueSelf(tspan, times, values, fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end
function ratefun(model::ChangedValueSelf)
    function(du, u, p, t)
        _, r₀, βr, kab_wt, kab_ko, nb_wt, nb_ko, pₛ, r₁, t₁ = p
        Q = u[1]; A = u[2]
        b = 1 - hill(t < t₁ ? kab_wt : kab_ko, 
                     t < t₁ ? nb_wt : nb_ko, Q+A)
        r = broken_exponential_decay(r₀, βr, r₁, t₁, t)
        du[1] = dQ = -r * Q + 2* b * pₛ * A
        du[2] = dA = r * Q - pₛ * A
    end
end
function initial(t::ChangedValueSelf, x::AbstractVector)
    nsc₀, r₀, _, kab_wt, _, nb_wt, _, pₛ,  _, _ = x
    b₀ = hill(kab_wt, nb_wt, 0)
    ratio = sqrt(((pₛ - r₀)/(2*r₀))^2 + (2*b₀*pₛ) / r₀)
    nsc₀ .* [1-1/(ratio+1), 1/(ratio+1)]
end
link(t::ChangedValueSelf, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]))
parameter_names(t::Type{ChangedValueSelf}) = [:nsc₀, :r₀, :βr, :kab_wt, :kab_ko, :nb_wt, :nb_ko, :pₛ, :r₁, :t₁]
bounds(t::ChangedValueSelf) = [(100.0, 10000.0), (0.0, 1.0), (0.0, 0.1), (0.0, 1000.0), (0.0, 1000.0), (0.01, 0.1), (0.01, 0.1), (0.0, 1.0), (0.0, 1.0), (0.0, 700.0)]
output_names(t::ChangedValueSelf) = ["qNSC", "aNSC"]
link_names(t::ChangedValueSelf) = ["Total NSC", "Fraction active NSC"]

#=
param_value = Dict( # from partial cofit
    :nsc₀ => 1986.0,
    :r₀ => 0.8426,
    :r₁ => 0.4928,
    :βr => 0.001891,
    :kab => 147.969,
    :nb_wt => 0.0300083,
    :nb_ko => 0.0765863,
    :pₛ => 0.95,
)
=#
param_value = Dict( # from partial cofit
    :nsc₀ => 3000.0,
    :r₀ => 0.80691,
    :r₁ => 0.493757,
    :βr => 0.00184341,
    :kab_wt => 148.364,
    :kab_ko => 155.445,
    :nb_wt => 0.0399102,
    :nb_ko => 0.0843197,
    :pₛ => 0.95,
)

times = collect(0:1:700.0)
valuemodel = ChangedValueSelf((0.0, 700.0), times, zeros(Int64, length(times)), param_value)

free_parameters(valuemodel)

valuemodel_r = t₁ -> broken_exponential_decay(param_value[:r₀], param_value[:βr], param_value[:r₁], t₁)

#pgfplots()
pyplot()
itimes = collect(0:1:660)
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
nonsim = link(valuemodel, rawsim)
sim = nonsim
ylabs = link_names(valuemodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab="Relative NSCs", xlab="Age (days)")
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
p6 = plot(ylab="Relative rate", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen_wt
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    #plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
    plot!(p6, times, neurogen ./ neurogen_wt, lab="", lc=colours[i], la=0.9)
    tot_rel[!,"d$(Int64(round(itime)))"] = sim[:,1] ./ tot_wt
    neurogen_rel[!,"d$(Int64(round(itime)))"] = neurogen ./ neurogen_wt
    tot_abs[!,"d$(Int64(round(itime)))"] = sim[:,1]
    neurogen_abs[!,"d$(Int64(round(itime)))"] = neurogen
end
# Plot previous simulation
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,1] ./ nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen_wt, lc=:black, lw=2, lab="")
plot!(p6, times, neurogen_wt ./ neurogen_wt, lc=:black, lw=2, lab="")
# Plot data
#@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
#@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p7 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
steps = round(minimum(total_neurogen); sigdigits=1):0.05:round(maximum(total_neurogen); sigdigits=1)
p8 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(100*x))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5 p6;
    p7 p8;
]
plot(p1, p2, p4, p3, p5, p6, p7, p8, lw=1, layout=l, size=(600, 800))
#savefig("jump_activation_individual_self.tex")
savefig("jump_activation_individual_self.svg")
tot_rel |> CSV.write("sims/jump_activation_individual_self_relative_nsc.csv")
neurogen_rel |> CSV.write("sims/jump_activation_individual_self_relative_neurogen.csv")
tot_abs |> CSV.write("sims/jump_activation_individual_self_absolute_nsc.csv")
neurogen_abs |> CSV.write("sims/jump_activation_individual_self_absolute_neurogen.csv")

#pgfplots()
pyplot()
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
itimes = collect(0:10:660)
nonsim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_beta
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
#p2 = plot(ylab=ylabs[2], xlab="Age (days)")
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    changed[max(1, findfirst(changed)-20):end] .= true
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times[changed], sim[changed,2], lab="", lc=colours[i], la=0.9)
#    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
end
# Plot previous sims
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,2], lab="", lc=:black, lw=2)
#p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
#plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen, lc=:black, lw=2,lab="")
# Plot data
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p6 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
neurogen_range = maximum(total_neurogen) - minimum(total_neurogen)
steps = 0.70:0.1:1.1
p7 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(round(100*x)))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5{.5h} [p6;
             p7]
]
plot(p1, p2, p4, p3, p5, p6, p7, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("jump_activation_individual_self.svg")

t₁ = 400.0
rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_value
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)
savefig("valuemodel_interaction.svg")

t₁ = 600.0
rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_value
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

t₁ = 30.0
rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_value
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

itimes = collect(0:10:699)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
p1 = plot(times, sim[:,1], lab="", lc=:gray, lw=2, yscale=:log10, ylab=ylabs[1])
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
#p2 = plot(times, sim[:,2], lab="", lc=:gray, lw=2, ylab=ylabs[2])
p3 = plot(times, param_rt[:wt], lab="", ylab="Activation rate", lc=:gray, lw=2)
#p4 = plot(times, param_bt[:wt], lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
p4 = plot(times, hillfun(800).(sim[:,1], times), lab="", 
          ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    sim = link(valuemodel, rawsim)
    has_changed = times .> t₁
    has_changed[max(1, findfirst(has_changed)-3):end] .= true
    plot!(p1, times[has_changed], sim[has_changed,1], lab="", lw=1, lc=colours[i]) #t₁=$t₁
    plot!(p2, times[has_changed], sim[has_changed,2], lab="", lw=1, lc=colours[i])
    plot!(p3, times[has_changed], valuemodel_r(t₁).(times[has_changed]), lab="", lw=1, lc=colours[i])
    plot!(p4, times[has_changed], hillfun(t₁).(sim[:,1], times)[has_changed], lab="", lw=1, lc=colours[i])
end
#plot!(p1, times, sim[:,1], lab="", lc=:red, lw=2)
@df @where(data, :genotype .== "wt", :name .== "total")  scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, p4, p3, xlab="Time (days)", size=(600, 600))

itimes = collect(0:1:660)#[100, 200]#, 60, 100, 300, 600]
itimes = metrics_loss.itime
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
regsim = link(valuemodel, rawsim)
nsc660 = regsim[times .== 660.0, 1][1]
nscs = map(itimes) do t₁
    rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
    regsim = link(valuemodel, rawsim)
    nsc660 .- regsim[times .== 660.0, 1][1]
end
metrics_loss[!,"value_fullself"] = nscs
plot(itimes, nscs, lc=:black, lab="", xlab="Intervention Time (days)", ylab="Stem cells lost", title="Loss of stem cells from intervention")

itimes = collect(0:10:600)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
sim = link(valuemodel, rawsim)
p = plot(xlab="Time (days)", ylab="Rate of TAP production from NSCs", yscale=:log10)
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    #has_changed = times .> t₁
    plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)), lab="", lc=colours[i])
end
rawsim = simulate(valuemodel, Dict(:t₁ => Float64(700)))
plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)), lab="WT", lc=:gray, lw=2)
#plot!(p, sim_ifnko_both.t, sim_ifnko_both.active .* sim_ifnko_both.counts .* sim_ifnko_both.b .* 2 .* 0.95, lab="KO", lc=:red, lw=2)
p
#savefig("plots/scenario1-lifelong-neurogenesis.svg")

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
#hillfun(t₁) = (x, t) -> 1 - hill(param[:kab], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
yticks = 50:5:115
metrics_neurogen[!,"value_fullself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=(collect(yticks), ["$x\\%" for x in yticks]))
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)

#=
param_value = Dict( # from cofit
    :nsc₀ => 1986.0,
    :r₀ => 0.8426,
    :r₁ => 0.4928,
    :βr => 0.001891,
    :kab => 141.987,
    :nb_wt => 0.0364225,
    :nb_ko => 0.0364225,
    #:nb_ko => 0.0765863,
    :pₛ => 0.95,
)
=#
param_value = Dict( # from cofit
    :nsc₀ => 3000.0,
    :r₀ => 0.839538,
    :r₁ => 0.491999,
    :βr => 0.00188605,
    :kab_wt => 153.81,
    :kab_ko => 153.81,
    :nb_wt => 0.039907,
    :nb_ko => 0.0841552,
    #:nb_ko => 0.0765863,
    :pₛ => 0.95,
)

times = collect(0:1:700.0) # 0.1 was step
valuemodel = ChangedValueSelf((0.0, 700.0), times, zeros(Int64, length(times)), param_value)

free_parameters(valuemodel)

valuemodel_r = t₁ -> broken_exponential_decay(param_value[:r₀], param_value[:βr], param_value[:r₁], t₁)

sim

#pgfplots()
pyplot()
itimes = collect(0:1:660)
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
nonsim = link(valuemodel, rawsim)
sim = nonsim
ylabs = link_names(valuemodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab="Relative NSCs", xlab="Age (days)")
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
p6 = plot(ylab="Relative rate", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen_wt
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    #plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
    plot!(p6, times, neurogen ./ neurogen_wt, lab="", lc=colours[i], la=0.9)
    tot_rel[!,"d$(Int64(round(itime)))"] = sim[:,1] ./ tot_wt
    neurogen_rel[!,"d$(Int64(round(itime)))"] = neurogen ./ neurogen_wt
    tot_abs[!,"d$(Int64(round(itime)))"] = sim[:,1]
    neurogen_abs[!,"d$(Int64(round(itime)))"] = neurogen
end
# Plot previous simulation
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,1] ./ nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen_wt, lc=:black, lw=2, lab="")
plot!(p6, times, neurogen_wt ./ neurogen_wt, lc=:black, lw=2, lab="")
# Plot data
#@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
#@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p7 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
steps = round(minimum(total_neurogen); sigdigits=1):0.05:round(maximum(total_neurogen); sigdigits=1)
p8 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(100*x))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5 p6;
    p7 p8;
]
plot(p1, p2, p4, p3, p5, p6, p7, p8, lw=1, layout=l, size=(600, 800))
#savefig("jump_activation_semishared_self.tex")
savefig("jump_activation_semishared_self.svg")
#tot_rel |> CSV.write("sims/jump_activation_semishared_self_relative_nsc.csv")
#neurogen_rel |> CSV.write("sims/jump_activation_semishared_self_relative_neurogen.csv")
#tot_abs |> CSV.write("sims/jump_activation_semishared_self_absolute_nsc.csv")
#neurogen_abs |> CSV.write("sims/jump_activation_semishared_self_absolute_neurogen.csv")

#pgfplots()
pyplot()
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
itimes = collect(0:10:660)
nonsim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
#p2 = plot(ylab=ylabs[2], xlab="Age (days)")
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    changed[max(1, findfirst(changed)-5):end] .= true
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen_tmp = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen_tmp .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times[changed], sim[changed,2], lab="", lc=colours[i], la=0.9)
#    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen_tmp[changed], lab="", lc=colours[i], la=0.9)
end
# Plot previous sims
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,2], lab="", lc=:black, lw=2)
#p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
#plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
#plot!(p3, times, param_rt[:wt], lab="", lc=:black, lw=2)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen, lc=:black, lw=2,lab="")
# Plot data
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p6 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
neurogen_range = maximum(total_neurogen) - minimum(total_neurogen)
steps = 0.80:0.05:1.1
p7 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(round(100*x)))\\%" for x in steps]), ylim=(minimum(steps), maximum(steps)))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5{.5h} [p6;
             p7]
]
plot(p1, p2, p4, p3, p5, p6, p7, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("jump_activation_semishared_self.svg")



total_mat = 0:10:660
#pgfplots()
pyplot()
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
itimes = collect(0:10:660)
all_nsc = zeros(length(times), length(itimes))
all_neurogen = zeros(length(times), length(itimes))
nonsim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
#p2 = plot(ylab=ylabs[2], xlab="Age (days)")
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    changed[max(1, findfirst(changed)-5):end] .= true
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen_tmp = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen_tmp .* step) ./ sum(non_neurogen)
    all_nsc[:,i] = sim[:,1]
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times[changed], sim[changed,2], lab="", lc=colours[i], la=0.9)
#    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen_tmp[changed], lab="", lc=colours[i], la=0.9)
    all_neurogen[:,i] = neurogen_tmp
end
# Plot previous sims
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,2], lab="", lc=:black, lw=2)
#p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
#plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
#plot!(p3, times, param_rt[:wt], lab="", lc=:black, lw=2)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen, lc=:black, lw=2,lab="")
# Plot data
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p6 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
neurogen_range = maximum(total_neurogen) - minimum(total_neurogen)
steps = 0.80:0.05:1.1
p7 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(round(100*x)))\\%" for x in steps]), ylim=(minimum(steps), maximum(steps)))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5{.5h} [p6;
             p7]
]
plot(p1, p2, p4, p3, p5, p6, p7, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("jump_activation_semishared_self.svg")

DataFrame("time" => itimes, "loss" => loss, "neurogen" => total_neurogen) |> CSV.write("data_export/all_metrics.csv")

neurogen

all_neurogen_df = DataFrame(all_neurogen, ["d$i" for i in itimes])
all_neurogen_df[!,"wt"] = neurogen
all_neurogen_df[!,"time"] = times

all_nsc_df = DataFrame(all_nsc, ["d$i" for i in itimes])
all_nsc_df[!,"wt"] = nonsim[:,1] 
all_nsc_df[!,"time"] = times

all_neurogen_df |> CSV.write("data_export/all_neurogen.csv")
all_nsc_df |> CSV.write("data_export/all_nsc.csv")

heatmap(log10.(all_nsc))

plot(neurogen, yscale=:log10)

itimes = collect(0:1:660)#[100, 200]#, 60, 100, 300, 600]
itimes = metrics_loss.itime
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
regsim = link(valuemodel, rawsim)
nsc660 = regsim[times .== 660.0, 1][1]
nscs = map(itimes) do t₁
    rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
    regsim = link(valuemodel, rawsim)
    nsc660 .- regsim[times .== 660.0, 1][1]
end
metrics_loss[!,"value_partself"] = nscs
plot(itimes, nscs, lc=:black, lab="", xlab="Intervention Time (days)", ylab="Stem cells lost", title="Loss of stem cells from intervention")

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
metrics_neurogen[!,"value_partself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=([100, 105, 110, 115], ["100\\%", "105\\%", "110\\%", "115\\%"]))
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)

#=
param_value = Dict( # from cofit
    :nsc₀ => 1986.0,
    :r₀ => 0.8426,
    :r₁ => 0.4928,
    :βr => 0.001891,
    :kab => 141.987,
    :nb_wt => 0.0364225,
    :nb_ko => 0.0364225,
    #:nb_ko => 0.0765863,
    :pₛ => 0.95,
)
=#
param_value = Dict( # from cofit
    :nsc₀ => 5000.0,
    :r₀ => 0.672387,
    :r₁ => 0.495324,
    :βr => 0.0015538,
    :kab_wt => 153.724,
    :kab_ko => 153.724,
    :nb_wt => 0.0584954,
    :nb_ko => 0.0584954,
    #:nb_ko => 0.0765863,
    :pₛ => 0.95,
)

times = collect(0:1:700.0)
valuemodel = ChangedValueSelf((0.0, 700.0), times, zeros(Int64, length(times)), param_value)

free_parameters(valuemodel)

valuemodel_r = t₁ -> broken_exponential_decay(param_value[:r₀], param_value[:βr], param_value[:r₁], t₁)

#pgfplots()
pyplot()
itimes = collect(0:1:660)
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
nonsim = link(valuemodel, rawsim)
sim = nonsim
ylabs = link_names(valuemodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(t < t₁ ? param[:kab_wt] : param[:kab_ko], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
p2 = plot(ylab="Relative NSCs", xlab="Age (days)")
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen_wt = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
tot_wt = sim[:,1]
tot_rel = DataFrame(:t => times, :wt => tot_wt ./ tot_wt)
neurogen_rel = DataFrame(:t => times, :wt => neurogen_wt ./ neurogen_wt)
tot_abs = DataFrame(:t => times, :wt => tot_wt)
neurogen_abs = DataFrame(:t => times, :wt => neurogen_wt)
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
p6 = plot(ylab="Relative rate", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen_wt
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times, sim[:,1] ./ tot_wt, lab="", lc=colours[i], la=0.9)
    #plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
    plot!(p6, times, neurogen ./ neurogen_wt, lab="", lc=colours[i], la=0.9)
    tot_rel[!,"d$(Int64(round(itime)))"] = sim[:,1] ./ tot_wt
    neurogen_rel[!,"d$(Int64(round(itime)))"] = neurogen ./ neurogen_wt
    tot_abs[!,"d$(Int64(round(itime)))"] = sim[:,1]
    neurogen_abs[!,"d$(Int64(round(itime)))"] = neurogen
end
# Plot previous simulation
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,1] ./ nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen_wt, lc=:black, lw=2, lab="")
plot!(p6, times, neurogen_wt ./ neurogen_wt, lc=:black, lw=2, lab="")
# Plot data
#@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
#@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p7 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
steps = round(minimum(total_neurogen); sigdigits=1):0.05:round(maximum(total_neurogen); sigdigits=1)
p8 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(100*x))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5 p6;
    p7 p8;
]
plot(p1, p2, p4, p3, p5, p6, p7, p8, lw=1, layout=l, size=(600, 800))
#savefig("jump_activation_shared_self.tex")
savefig("jump_activation_shared_self.svg")
tot_rel |> CSV.write("sims/jump_activation_shared_self_relative_nsc.csv")
neurogen_rel |> CSV.write("sims/jump_activation_shared_self_relative_neurogen.csv")
tot_abs |> CSV.write("sims/jump_activation_shared_self_absolute_nsc.csv")
neurogen_abs |> CSV.write("sims/jump_activation_shared_self_absolute_neurogen.csv")

#pgfplots()
pyplot()
rawsim = simulate(valuemodel, Dict(:t₁ => 800.0)) # No intervention model
itimes = collect(0:10:660)
nonsim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_beta
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
# Plot base simulation (wt)
p1 = plot(yscale=:log10, ylab=ylabs[1], xlab="Age (days)")
#p2 = plot(ylab=ylabs[2], xlab="Age (days)")
p2 = plot(ylab=ylabs[2], xlab="Age (days)", ylim=(0.18, 0.65))
p3 = plot(xlab="Age (days)", ylab="Activation rate")
p4 = plot(xlab="Age (days)",  ylab="Self-renewal probability", legend=:bottomright)
# ???
neurogen = 2 .* 0.95 .* (1 .- hillfun(800.0).(sim[:,1], times)) .* rawsim[:,2]
p5 = plot(yscale=:log10, ylab="Rate of progenitor production", xlab="Age (days)")
# Plot interventions
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
# Compute some things
step = times[2] - times[1]
non_neurogen = step .* neurogen
# Preallocate vectors
loss = zeros(Float64, length(itimes))
total_neurogen = zeros(Float64, length(itimes))
for (i, itime) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(itime)))
    changed = times .> itime
    changed[max(1, findfirst(changed)-5):end] .= true
    sim = link(valuemodel, rawsim)
    loss[i] = nonsim[findfirst(times .== 660.0), 1] .- sim[findfirst(times .== 660.0),1]
    neurogen = 2 .* 0.95 .* (1 .- hillfun(itime).(sim[:,1], times)) .* rawsim[:,2]
    total_neurogen[i] = sum(neurogen .* step) ./ sum(non_neurogen)
    plot!(p1, times[changed], sim[changed,1], lab="", lc=colours[i], la=0.9)
    plot!(p2, times[changed], sim[changed,2], lab="", lc=colours[i], la=0.9)
#    plot!(p3, times[changed], rawsim[changed,3], lab="", lc=colours[i], la=0.9)
    plot!(p3, times[changed], valuemodel_r(itime).(times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p4, times[changed], hillfun(itime).(sim[changed,1], times[changed]), lab="", lc=colours[i], la=0.9)
    plot!(p5, times[changed], neurogen[changed], lab="", lc=colours[i], la=0.9)
end
# Plot previous sims
plot!(p1, times, nonsim[:,1], lab="", lc=:black, lw=2)
plot!(p2, times, nonsim[:,2], lab="", lc=:black, lw=2)
#p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
#plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
#plot!(p3, times, param_rt[:wt], lab="", lc=:black, lw=2)
plot!(p3, times, valuemodel_r(900.0), lab="", lc=:black, lw=2)
plot!(p4, times, hillfun(800.0).(nonsim[:,1], times), lab="", lc=:black, lw=2)
plot!(p5, times, neurogen, lc=:black, lw=2,lab="")
# Plot data
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
# Plot metrics
p6 = plot(itimes, loss, lab="", ylab="Stem cells lost", xlab="Intervention time (days)", lc=:black)
neurogen_range = maximum(total_neurogen) - minimum(total_neurogen)
steps = 1.00:0.05:1.1
p7 = plot(itimes, total_neurogen, lc=:black, lab="", ylab="Total progenitor gain", xlab = "Intervention time (days)",
          ytick=(collect(steps), ["$(Int64(round(100*x)))\\%" for x in steps]))
#p7 = plot(itimes, total_neurogen)
l = @layout [
    p1 p2;
    p3 p4;
    p5{.5h} [p6;
             p7]
]
plot(p1, p2, p4, p3, p5, p6, p7, lw=1, layout=l, size=(600, 800))
#savefig("freeze_activation_individual_self.tex")
savefig("jump_activation_shared_self.svg")

t₁ = 400.0
rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_value
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)
savefig("valuemodel_static.svg")

t₁ = 600.0
rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_value
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

t₁ = 30.0
rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
param = param_value
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
p1 = plot(times, sim[:,1], lab="", lc=:red, lw=2, yscale=:log10, ylab=ylabs[1])
@df @where(data, :genotype .== "wt", :name .== "total") scatter!(p1, :age, :value, mc=:black, lab="")
p2 = plot(times, sim[:,2], lab="", lc=:red, lw=2, ylab=ylabs[2])
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
p3 = plot(times, param_rt[:wt], lab="wt", lc=:black)
plot!(p3, times, valuemodel_r(t₁).(times), lab="del", ylab="Activation rate", lc=:red)
p4 = plot(times, hillfun(800.0).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black)
plot!(p4, times, hillfun(t₁).(sim[:,1], times), lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:red)
plot(p1, p2, p4, p3, xlab="Time (days)", lw=2)

itimes = collect(0:10:699)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
sim = link(valuemodel, rawsim)
ylabs = link_names(valuemodel)
p1 = plot(times, sim[:,1], lab="", lc=:gray, lw=2, yscale=:log10, ylab=ylabs[1])
p2 = plot(times, sim[:,2], lab="", lc=:gray, lw=2, ylab=ylabs[2])
p3 = plot(times, param_rt[:wt], lab="", ylab="Activation rate", lc=:gray, lw=2)
#p4 = plot(times, param_bt[:wt], lab="", ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
p4 = plot(times, hillfun(800).(sim[:,1], times), lab="", 
          ylab="Self-renewal probability", legend=:bottomright, lc=:black, lw=2)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    sim = link(valuemodel, rawsim)
    has_changed = times .> t₁
    has_changed[max(1, findfirst(has_changed)-3):end] .= true
    plot!(p1, times[has_changed], sim[has_changed,1], lab="", lw=1, lc=colours[i]) #t₁=$t₁
    plot!(p2, times[has_changed], sim[has_changed,2], lab="", lw=1, lc=colours[i])
    plot!(p3, times[has_changed], valuemodel_r(t₁).(times[has_changed]), lab="", lw=1, lc=colours[i])
    plot!(p4, times[has_changed], hillfun(t₁).(sim[:,1], times)[has_changed], lab="", lw=1, lc=colours[i])
end
#plot!(p1, times, sim[:,1], lab="", lc=:red, lw=2)
@df @where(data, :genotype .== "wt", :name .== "total")  scatter!(p1, :age, :value, mc=:black, lab="")
@df @where(data, :genotype .== "wt", :name .== "active") scatter!(p2, :age, :value, mc=:black, lab="")
plot(p1, p2, p4, p3, xlab="Time (days)", size=(600, 600))

itimes = collect(0:1:660)#[100, 200]#, 60, 100, 300, 600]
itimes = metrics_loss.itime
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
regsim = link(valuemodel, rawsim)
nsc660 = regsim[times .== 660.0, 1][1]
nscs = map(itimes) do t₁
    rawsim = simulate(valuemodel, Dict(:t₁ => t₁))
    regsim = link(valuemodel, rawsim)
    nsc660 .- regsim[times .== 660.0, 1][1]
end
metrics_loss[!,"value_noself"] = nscs
plot(itimes, nscs, lc=:black, lab="", xlab="Intervention Time (days)", ylab="Stem cells lost", title="Loss of stem cells from intervention")

itimes = collect(0:10:600)#[100, 200]#, 60, 100, 300, 600]
colours = cgrad(:roma, length(itimes), categorical=true, rev=false)
rawsim = simulate(valuemodel, Dict(:t₁ => 700.0))
sim = link(valuemodel, rawsim)
p = plot(xlab="Time (days)", ylab="Rate of TAP production from NSCs", yscale=:log10)
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    #has_changed = times .> t₁
    plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)), lab="", lc=colours[i])
end
rawsim = simulate(valuemodel, Dict(:t₁ => Float64(700)))
plot!(p, times, rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)), lab="WT", lc=:gray, lw=2)
#plot!(p, sim_ifnko_both.t, sim_ifnko_both.active .* sim_ifnko_both.counts .* sim_ifnko_both.b .* 2 .* 0.95, lab="KO", lc=:red, lw=2)
p
#savefig("plots/scenario1-lifelong-neurogenesis.svg")

itimes = metrics_loss.itime
int_neurogenesis_1 = zeros(Float64, length(itimes))
#int_neurogenesis_2 = zeros(Float64, length(itimes))
int_neurogenesis_wt = zeros(Float64, length(itimes))
step = times[2] - times[1]
hillfun(t₁) = (x, t) -> 1 - hill(param[:kab_wt], t < t₁ ? param[:nb_wt] : param[:nb_ko])(x)
for (i, t₁) in enumerate(itimes)
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    int_neurogenesis_1[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
    #rawsim = simulate(valuemodel, Dict(:t₁ => Float64(t₁)))
    #int_neurogenesis_2[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- param_bt[:wt].(times)))
    rawsim = simulate(valuemodel, Dict(:t₁ => Float64(700)))
    int_neurogenesis_wt[i] = sum(rawsim[:,2] .* 2 .* 0.95 .* (1 .- hillfun(t₁).(rawsim[:,1], times)))
end
metrics_neurogen[!,"value_noself"] = int_neurogenesis_1 ./ int_neurogenesis_wt
plot(xlab="Intervention Time (days)", ylab="increase in TAPs produced over wt", title="Life-long (700 days) neurogenesis", yticks=([100, 105, 110, 115], ["100\\%", "105\\%", "110\\%", "115\\%"]))
plot!(itimes, int_neurogenesis_1 ./ int_neurogenesis_wt .* 100, lab="'beta' model")
#plot!(itimes, int_neurogenesis_2 ./ int_neurogenesis_wt .* 100, lab="'value' model")
#plot!(itimes, int_neurogenesis_wt .* step, lab="wt", lc=:gray)



names(metrics_neurogen)

ticks = 0.9:0.05:1.07
p = plot(legend=:bottomright, 
         ylim=(0.9, 1.07), 
         xlim=(0, 700),
         size=(300, 200),
         ytick=(ticks, ["$(round(Int64, 100*x))\\%" for x in ticks]),
         xlab="Time (days)", ylab="TAPs produced over wt")
Plots.abline!(0, 1, lab="", lc=:black)
#@df metrics_neurogen plot!(:itime, :beta_selfinteraction, lab="`beta` model with interaction", lw=2, lc=1)
#@df metrics_neurogen plot!(:itime, :beta_selfinvariant, lab="`beta` model without interaction", lw=2, lc=2)
@df metrics_neurogen plot!(:itime, :value_partself, lab="IFN dependent self-renewal", lw=2, lc=1)
@df metrics_neurogen plot!(:itime, :value_noself, lab="IFN independent self-renewal", lw=2, lc=2)
savefig("plot/life-long-pop-self.svg")

neurogen = @linq metrics_neurogen |>
    select(:t = :itime, :neurogen_dependent = :value_partself, :neurogen_independent = :value_noself)
loss = @linq metrics_loss |> 
    select(:t = :itime, :loss_dependent = :value_partself, :loss_independent = :value_noself)
innerjoin(neurogen, loss, on="t") |> CSV.write("metrics.csv")

p = plot(legend=:topleft, xlab="Time (days)", ylab="Stem cells lost at 660 days",
         size=(300, 300),
         xlim=(0, 700), ylim=(0, 10))
#@df metrics_loss plot!(:itime, :beta_selfinteraction, lab="`beta` model with interaction", lw=2)
#@df metrics_loss plot!(:itime, :beta_selfinvariant, lab="`beta` model without interaction", lw=2)
#@df metrics_loss plot!(:itime, :value_selfinteraction, lab="Independent self-renewal", lw=2)
#@df metrics_loss plot!(:itime, :value_selfinvariant, lab="Combined self-renewal", lw=2)
@df metrics_loss plot!(:itime, :value_partself, lab="IFN dependent self-renewal", lw=2, lc=1)
@df metrics_loss plot!(:itime, :value_noself, lab="IFN independent self-renewal", lw=2, lc=2)
savefig("plot/loss-pop-self.svg")































struct ChangedBetaSelf <: ODEModel
    tspan::Tuple{Float64, Float64}
    times::Vector{Float64}
    values::Vector{Int64}
    fixed::Vector{Pair{Int64, Float64}}
    ChangedBetaSelf(tspan, times, values, fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, times, values, fixed)
end
function ratefun(model::ChangedBetaSelf)
    function(du, u, p, t)
        _, _, βr, kab, nb_wt, nb_ko, pₛ, t₁ = p
        Q = u[1]; A = u[2]; r = u[3]
        b = 1 - hill(kab, t < t₁ ? nb_wt : nb_ko, Q+A)
        du[1] = dQ = -r * Q + 2* b * pₛ * A
        du[2] = dA = r * Q - pₛ * A
        if t < t₁
            du[3] = dr = -βr * r
        else
            du[3] = dr = 0
        end
    end
end
function initial(t::ChangedBetaSelf, x::AbstractVector)
    nsc₀, r₀, _, kab, nb_wt, _, pₛ, _ = x
    b₀ = hill(kab, nb_wt, 0)
    ratio = sqrt(((pₛ - r₀)/(2*r₀))^2 + (2*b₀*pₛ) / r₀)
    vcat(nsc₀ .* [1-1/(ratio+1), 1/(ratio+1)], [r₀])
end
link(t::ChangedBetaSelf, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]))
parameter_names(t::Type{ChangedBetaSelf}) = [:nsc₀, :r₀, :βr, :kab, :nb_wt, :nb_ko, :pₛ, :t₁]
bounds(t::ChangedBetaSelf) = [(100.0, 10000.0), (0.0, 1.0), (0.0, 0.1), (0.0, 1000.0), (0.01, 0.1), (0.01, 0.1), (0.0, 1.0), (0.0, 700.0)]
output_names(t::ChangedBetaSelf) = ["qNSC", "aNSC", "r"]
link_names(t::ChangedBetaSelf) = ["Total NSC", "Fraction active NSC"]
