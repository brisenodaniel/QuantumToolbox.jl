# Replicate the plots in Qutip Floquet tutorial
using CairoMakie
using LinearAlgebra

## the functions below work only for vectors not for density matrices

function floquet_mode(fb::FloquetBasis, t::Float64)
    #t = mod(t, fb.T)
    eigs = eigenstates(propagator(fb, 0.0, fb.T))
    evecs_mat = eigs.vectors  
    phases = Diagonal(exp.(1im * t .* fb.equasi))
    return propagator(fb, 0.0, t).data * evecs_mat * phases
end

function floquet_state(fb::FloquetBasis, t::Real; data::Bool=false)
    mode_mat = floquet_mode(fb, t)
    return  mode_mat * Diagonal(exp.(-1im * t .* fb.equasi))
end

function to_floquet_basis(
    fb::FloquetBasis,
    ψ::QuantumObject{Ket},
    t::Float64,
    )
return (floquet_state(fb, t)' |> Qobj) * ψ
end

function from_floquet_basis(
    fb::FloquetBasis,
    ψ::QuantumObject{Ket}, # floquet state
    t::Float64,
    )
return (floquet_state(fb, t) |> Qobj) * ψ
end


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
f_energies = fb1.equasi
f_modes_1 = floquet_mode(fb1, 0.0)



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

#fig |> display
    
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
#fig3 |> display

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
#fig4 |> display


### Tests 
N = 10
a = destroy(N)
a_d = a'
H = num(N) + (a + a_d)
Ht = QobjEvo(H, (p, t) -> cos(t)) # for test throw
T = 2π
psi0 = fock(N, 3)
t_l = LinRange(0, 200, 1000)
fb_test1 = FloquetBasis(Ht, T)



#sol_me = mesolve(H, psi0, t_l, c_ops, e_ops = e_ops, progress_bar = Val(false))
#rho_me = sol_me.states[end]









