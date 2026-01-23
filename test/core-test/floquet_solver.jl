using CairoMakie
using LinearAlgebra

N = 90 # Hilbert space dimension
ω0 = 1 # oscillator frequency as the unit
g3 = 7.5e-4 #third order nonlinearity
g4 = 4.027e-6 # fourth order nonlinearity
ωd = 2.0 # two-photon drive frequency
T  = 4π / ωd # twice the usual period because of two-photon drive

M = 0.001
Ω_d = LinRange(0.0,M,10) # drive amplitude

K  = -3 * g4 / 2 + 10 * g3^2 / (3 * ωd) # effective Kerr nonlinearity
a = destroy(N)
a_d = a'

q_energies = zeros(Float64, length(Ω_d), N)

ad3 = (a + a_d)^3
ad4 = (a + a_d)^4

for (idx, Omd) in enumerate(Ω_d)
    H0 = ω0 * a_d * a +(g3 / 3) * ad3 + (g4 / 4) * ad4 
    H1 = -im * Omd * (a - a_d)
    f(p, t) = cos(ωd * t)
    Hevo = (H0, (H1, f)) |> QobjEvo
    fbasis = FloquetBasis(Hevo, T)
    q_energies[idx, :] .= sort(fbasis.equasi; rev = true)
end




fig1 = Figure()
ax1 = Axis(fig1[1, 1],
    xlabel = L"\Omega_d",
    ylabel = L"(\epsilon_n - \epsilon_0)/K"
)

lines!(ax1, Ω_d, (q_energies[:, 2] .- q_energies[:, 1]) ./ K)
lines!(ax1, Ω_d, (q_energies[:, 3] .- q_energies[:, 1]) ./ K)
lines!(ax1, Ω_d, (q_energies[:, 4] .- q_energies[:, 1]) ./ K)
lines!(ax1, Ω_d, (q_energies[:, 5] .- q_energies[:, 1]) ./ K)
lines!(ax1, Ω_d, (q_energies[:, 6] .- q_energies[:, 1]) ./ K)
lines!(ax1, Ω_d, (q_energies[:, 7] .- q_energies[:, 1]) ./ K)
lines!(ax1, Ω_d, (q_energies[:, 8] .- q_energies[:, 1]) ./ K)

fig1 |> display

eigs = zeros(Float64, length(Ω_d), N)
for (idx,Omd) in enumerate(Ω_d)
    Π = 4 * Omd / (3 * ωd)
    Δ = ω0 - ωd/2 + 6 * g4 * Π^2 - 18 * g3^2 * Π^2 / ωd + 2 * K
    ϵ2 = g3 * Π
    Ham_K = Δ * a_d * a - (K / 2) * a_d * a_d * a * a + ϵ2 * (a_d * a_d + a * a)
    eigs[idx, :] .= real.(eigenenergies(Ham_K))
end

fig = Figure()
ax = Axis(fig[1, 1],
    xlabel = L"\Omega_d",
    ylabel = L"(\epsilon_n - \epsilon_0)/K"
)

lines!(ax, Ω_d, -(eigs[:, 2] .- eigs[:, 1]) ./ K)
lines!(ax, Ω_d, -(eigs[:, 3] .- eigs[:, 1]) ./ K)
lines!(ax, Ω_d, -(eigs[:, 4] .- eigs[:, 1]) ./ K)
lines!(ax, Ω_d, -(eigs[:, 5] .- eigs[:, 1]) ./ K)
lines!(ax, Ω_d, -(eigs[:, 6] .- eigs[:, 1]) ./ K)
lines!(ax, Ω_d, -(eigs[:, 7] .- eigs[:, 1]) ./ K)
lines!(ax, Ω_d, -(eigs[:, 8] .- eigs[:, 1]) ./ K)

fig |> display