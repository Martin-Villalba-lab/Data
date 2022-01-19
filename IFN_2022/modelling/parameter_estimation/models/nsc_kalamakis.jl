
struct KalamakisFull <: ODEModel
    tspan::Tuple{Float64, Float64}
    fixed::Vector{Pair{Int64, Float64}} # pointer to thing to fix and value
    KalamakisFull(tspan, fixed::Vector{Pair{I, N}}) where {I <: Integer, N <: Number} = new(tspan, fixed)
end

function ratefun(model::KalamakisFull)
    function(du, u, p, t)
        _, r₀, βr, b₀, βb, pₛ = p
        Q = u[1]; A = u[2]
        du[1] = dQ = -exponential_decay(r₀, βr, t) * Q + 2 * flattening_curve(b₀, βb, t) * pₛ * A
        du[2] = dA =  exponential_decay(r₀, βr, t) * Q - pₛ * A
    end
end

function initial(t::KalamakisFull, x::AbstractVector)
    nsc₀, r₀, _, b₀, _, pₛ = x
    ratio = sqrt(((pₛ - r₀)/(2*r₀))^2 + (2*b₀*pₛ) / r₀)
    nsc₀ .* [1-1/(ratio+1), 1/(ratio+1)]
end

link(t::KalamakisFull, x::AbstractArray) = hcat(x[:,1] .+ x[:,2], x[:,2] ./ (x[:,1] .+ x[:,2]))

parameter_names(t::Type{KalamakisFull}) = [:nsc₀, :r₀, :βr, :b₀, :βb, :pₛ]

bounds(t::KalamakisFull) = [(100.0, 2500.0), (0.0, 1.0), (0.0, 0.1), (0.0, 0.5), (0.0, 0.1), (0.0, 1.0)]

link_styles(t::KalamakisFull) = [:log10, :identity]

output_names(t::KalamakisFull) = ["qNSC", "aNSC"]
link_names(t::KalamakisFull) = ["Total NSC", "Fraction active NSC"]
