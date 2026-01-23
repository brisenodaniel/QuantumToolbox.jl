# Replicate the plots in Qutip Floquet tutorial
using CairoMakie
using LinearAlgebra
@doc raw"""
Let us consider a driven two-level system with the Hamiltonian
```math
H(t) = -\frac{1}{2}\Delta \sigma_x - \frac{1}{2}\epsilon_0 \sigma_z + \frac{1}{2}A \sin(\omega t) \sigma_z
```
where ``A`` is the driving amplitude and ``\omega`` is the driving frequency. In QuantumToolbox, the Hamiltonian 
above can be used to construct a Floquet basis as follows:

"""


const σx = sigmax()
const σz = sigmaz()

H0(; δ, ϵ0, kwargs...) = -δ/2 * σx - ϵ0/2 * σz
H1x(; A, kwargs...) = A/2.0 * σx
H1z(; A, kwargs...) = A/2.0 * σz
f(param) = (p, t) -> sin(param.ω * t)


Hevoz(param::NamedTuple) = (H0(; param...), (H1z(; param...), f(param))) |> QobjEvo
                           
Hevox(param::NamedTuple) = (H0(; param...), (H1x(; param...), f(param))) |> QobjEvo
                    
# First box
par1 = (
    ϵ0 = 1.0 * 2π,
    δ  = 0.2 * 2π,
    A  = 2.5 * 2π,
    ω  = 1.0 * 2π
)
T = 2π / par1.ω
fb1 = FloquetBasis(Hevoz(par1),T)

@doc raw"""
The floquet quasienergies and the floquet modes at time ``t=0`` can be obtained as follows:
"""

f_energies = fb1.equasi
f_modes_1 = modes(fb1, t = 0.0)


@doc raw"""
Now we are ready to study the floquet quasienergies as a function of the driving amplitude ``A``. Quasienergy levels
can provide insight into system dynamics. For a driven two-level system, plotting quasienergies against the driving amplitude
reveals crossings at specific amplitudes. These degeneracies correspond to a suppression of dynamics, 
a phenomenon known as coherent destruction of tunneling.
"""

#Second box
A_vec = range(0.0, 10.0; length=100) .* par1.ω
tlist = range(0.0, 10.0 * T; length=101)
q_energies = zeros(length(A_vec),2)
for (i,A_val) in enumerate(A_vec)
    par2 = (ϵ0 = 0.0, δ = par1.δ , A = A_val, ω = par1.ω)
    q_energies[i,:] .= FloquetBasis(Hevoz(par2),T).equasi
end

fig = Figure(size = (500, 350))
ax = Axis(fig[1,1], xlabel = L"A/$\omega$", ylabel = L"Quasienergy/$\Delta$", title = "Floquet Quasienergies")
lines!(ax, A_vec ./ par1.ω, q_energies[:,1] ./ par1.δ, color = :blue)
lines!(ax, A_vec ./ par1.ω, q_energies[:,2] ./ par1.δ, color = :red)

fig |> display

@doc raw """
discuss what to and from floquet basis actually do..... and how you can just use fsesolve

"""
    
#Third box
par3 = (ϵ0 = par1.ϵ0, δ = par1.δ, A = 0.5 * 2 * π, ω = par1.ω)
psi0 = basis(2,0)

fb3 = FloquetBasis(Hevoz(par3),T)
f_coef3 = to_floquet_basis(fb3, psi0, 0.0)
p_ex3 = [real.(expect(num(2), from_floquet_basis(fb3, f_coef3, t))) for t in tlist]

solution3 = sesolve(Hevoz(par3), psi0, tlist)
p_ex_ref3 = expect(num(2), solution3.states)


fig3 = Figure(size = (500, 350))
ax3 = Axis(fig3[1,1], xlabel = L"Time$(t)$", ylabel = L"Occupation Probability $$", title = "Population")

lines!(ax3, tlist, p_ex3, color = :blue, label=L"Floquet $P_1$")
scatter!(ax3, tlist, real.(p_ex3), color=:red)

lines!(ax3, tlist, 1 .-p_ex3, color = :red, label=L"Floquet $P_0$")
scatter!(ax3, tlist, 1 .- real.(p_ex3), color=:blue)

lines!(ax3, tlist, real.(p_ex_ref3), color=:red, linestyle=:dash, label=L"Lindblad $P_1$")
lines!(ax3, tlist, 1 .- real.(p_ex_ref3), color=:blue, linestyle=:dash, label=L"Lindblad $P_0$")

axislegend(ax3, position=:rt, outside=true)
fig3 |> display

#Fourth box
par4 = (ϵ0 = par1.ϵ0, δ = 0.0, A = 0.25 * 2 * π, ω = par1.ω)

fb4 = FloquetBasis(Hevox(par4),T)


f_coef4 = to_floquet_basis(fb4, psi0, 0.0)
p_ex4 = [real.(expect(num(2), from_floquet_basis(fb4, f_coef4, t))) for t in tlist]

solution4 = sesolve(Hevox(par4), psi0, tlist)
p_ex_ref4 = expect(num(2), solution4.states)

fig4 = Figure(size = (500, 350))
ax4 = Axis(fig4[1,1], xlabel = L"Time$(t)$", ylabel = L"Occupation Probability $$", title = "Population")

lines!(ax4, tlist, p_ex4, color = :blue, label=L"Floquet $P_1$")
scatter!(ax4, tlist, real.(p_ex4), color=:red)

lines!(ax4, tlist, 1 .-p_ex4, color = :red, label=L"Floquet $P_0$")
scatter!(ax4, tlist, 1 .- real.(p_ex4), color=:blue)

lines!(ax4, tlist, real.(p_ex_ref4), color=:red, linestyle=:dash, label=L"Lindblad $P_1$")
lines!(ax4, tlist, 1 .- real.(p_ex_ref4), color=:blue, linestyle=:dash, label=L"Lindblad $P_0$")

axislegend(ax4, position=:rt, outside=true)
fig4 |> display










