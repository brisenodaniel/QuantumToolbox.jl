# Replicate the plots in Qutip Floquet tutorial
using CairoMakie
using LinearAlgebra

struct Params
    ϵ0::Float64
    δ::Float64
    A::Float64
    ω::Float64
end

function H0(params::Params)
    return - params.δ/2.0 * sigmax() - params.ϵ0/2.0 * sigmaz()
end

function H1(params::Params)
    return params.A/2.0 * sigmaz()
end

function Hevo(params::Params)
    f(p,t) = cos(params.ω * t)
    H_tuple = (H0(params), (H1(params), f))
    return QobjEvo(H_tuple)
end

#First box
par1 = Params(1.0 * 2 * π, 0.2 * 2 * π, 2.5 * 2 * π, 1.0 * 2 * π)
T = 2π / par1.ω
floquet_basis = FloquetBasis(Hevo(par1),T)
f_energies = floquet_basis.equasi
#f_modes_1 = mode(floquet_basis, t = 0.1 * T)




# Second box
ω2 = 1.0 * 2 * π
δ2 = 0.2 * 2 * π
T2 = 2π / ω2
A_vec = range(0.0,10.0,100) * ω2
tlist = range(0.0, 10*T2, 101)
q_energies = zeros(length(A_vec),2)
for (i,A_val) in enumerate(A_vec)
    par2 = Params(0.0 * 2 * π, δ2, A_val, ω2)
    q_energies[i,:] = FloquetBasis(Hevo(par2),T2).equasi
end

fig = Figure(size = (500, 350))
ax = Axis(fig[1,1], xlabel = L"A/$\omega$", ylabel = L"Quasienergy/$\Delta$", title = "Floquet Quasienergies")
lines!(ax, A_vec ./ ω2, q_energies[:,1] ./ δ2, color = :blue)
lines!(ax, A_vec ./ ω2, q_energies[:,2] ./ δ2, color = :red)

fig |> display
    
#Third box
par2 = Params(1.0 * 2 * π, 0.2 * 2 * π, 0.5 * 2 * π, 1.0 * 2 * π)
T3 = 2π / par2.ω
tlist3 = range(0.0, 10*T3, 101) 
psi0 = basis(2,1)

floquet_basis3 = FloquetBasis(Hevo(par2),T3)

## the functions below work only for vectors not for density matrices

function floquet_mode(fb::FloquetBasis, t::Float64)
    eigs = eigenstates(propagator(fb, 0.0, fb.T))
    evecs_mat = eigs.vectors  
    phases = Diagonal(exp.(-1im * t .* fb.equasi))
    return propagator(fb, t, fb.T).data * evecs_mat * phases
end

function floquet_state(fb::FloquetBasis, t::Real; data::Bool=false)

    mode_mat = floquet_mode(fb, t)
    return Diagonal(exp.(-1im * t .* fb.equasi)) * mode_mat
end

function to_floquet_basis(
    fb::FloquetBasis,
    ψ::QuantumObject{Ket},
    )
return (floquet_state(fb, 0.0)' |> Qobj) * ψ
end












