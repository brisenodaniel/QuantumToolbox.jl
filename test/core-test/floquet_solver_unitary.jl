using QuantumToolbox
using Test 
using Random
using CarioMakie




@testitem "FloquetBasis" begin
    σz, σx, σy = (sigmaz(), sigmax(), sigmay())
    H0(;δ, ϵ0, kwargs...) = -(δ/2)*σx - (ϵ0/2)*σz
    H1(;A, kwargs...) = (A/2) * σz
    d(p, t) = cos(p.ω * t)

    # box 1
    b1_p = (ϵ0=2π, δ=0.2*2π, A=2.5*2π, ω=2π)
    T_b1 = 2π/b1_p[:ω]
    H_b1 = (H0(;b1_p...), (H1(;b1_p...), d)) |> QobjEvo
    fb_b1 = FloquetBasis(H_b1, T_b1; params=b1_p)

    f_energies = fb_b1.equasi

end



@testitem "Floquet Solver" begin
    N = 10
    a = destroy(N)
    a_d = a'
    coef(p,t) = cos(t)
    H0 = num(N)
    H1 = a + a_d
    H_tuple = (H0,(H1,coef))
    H_evo = QobjEvo(H_tuple)
    T = 2 * π
    tlist = range(0.0, 3T, length=101)
    floquet_basis = FloquetBasis(H_evo, T, tlist)
    psi0 = rand_ket(N)
    floquet_psi0 = to_floquet_basis(floquet_basis, psi0)
    sol = sesolve(H_tuple, psi0, tlist, e_ops = [], saveat = tlist)
    states = sol.states
    fse = floquet_sesolve(floquet_basis, psi0, tlist, T=T)
    states_fse = fse.states
    for (t,state) in zip(tlist,states)
        from_floquet = from_floquet_basis(floquet_basis, floquet_psi0, t)
        @test overlap(state, from_floquet) ≈ 1.0 atol=8e-5  
    end
    for (state_se,state_fse) in zip(states, states_fse)
        @test overlap(state_se, state_fse) ≈ 1.0 atol=5e-5  
    end    
end






