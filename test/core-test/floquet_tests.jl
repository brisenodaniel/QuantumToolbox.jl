using LinearAlgebra
using Test

#Test comparing the output of sesolve.states with the from_floquet_basis and fsesolve

# N = 2 with Qutip tolerance 8e-5 for from floquet_basis and 5e-5 for fsesolve

# same system as considered in test_floquet.py in qutip

@testitem "Test Floquet Basis1" begin
    N = 2     
    a = destroy(N)
    a_d = a'
    H = num(N) + (a + a_d)
    Ht = QobjEvo(H, (p, t) -> cos(t)) # For test, consider ω = 1
    T = 2π
    psi0 = rand_ket(N)
    t_l = LinRange(0, 200, 1000)
    fb_test1 = FloquetBasis(Ht, T)
    floquet_psi0 = to_floquet_basis(fb_test1, psi0)
    states_se = sesolve(Ht, psi0, t_l).states
    states_fse = fsesolve(fb_test1, psi0, t_l).states

    # Test overlap between from_floquet_basis and sesolve states
    for (t, state) in zip(t_l, states_se)
        from_floquet = from_floquet_basis(fb_test1, floquet_psi0, t)
        ov = abs(state' * from_floquet)
        @test isapprox(ov, 1.0; atol=2e-3)
    end

    # Test overlap between fsesolve and sesolve states
    for (state_s, state_f) in zip(states_se, states_fse)
        ov = abs(state_s' * state_f)
        @test isapprox(ov, 1.0; atol=5e-5) 
    end
end

# N = 10 with Qutip tolerance 8e-5 for from floquet_basis and 5e-5 for fsesolve

@testitem "Test Floquet Basis2" begin
    N = 10     
    a = destroy(N)
    a_d = a'
    H = num(N) + (a + a_d)
    Ht = QobjEvo(H, (p, t) -> cos(t)) # For test, consider ω = 1
    T = 2π
    psi0 = rand_ket(N)
    t_l = LinRange(0, 200, 1000)
    fb_test1 = FloquetBasis(Ht, T)
    floquet_psi0 = to_floquet_basis(fb_test1, psi0)
    states_se = sesolve(Ht, psi0, t_l).states
    states_fse = fsesolve(fb_test1, psi0, t_l).states

    # Test overlap between from_floquet_basis and sesolve states
    for (t, state) in zip(t_l, states_se)
        from_floquet = from_floquet_basis(fb_test1, floquet_psi0, t)
        ov = abs(state' * from_floquet)
        @test isapprox(ov, 1.0; atol=8e-5)
    end

    # Test overlap between fsesolve and sesolve states
    for (state_s, state_f) in zip(states_se, states_fse)
        ov = abs(state_s' * state_f)
        @test isapprox(ov, 1.0; atol=5e-5) 
    end
end

# N = 10 with lower maximum allowed tolerance 2e-3 for from floquet_basis:

@testitem "Test Floquet Basis3" begin
    N = 10     
    a = destroy(N)
    a_d = a'
    H = num(N) + (a + a_d)
    Ht = QobjEvo(H, (p, t) -> cos(t)) # For test, consider ω = 1
    T = 2π
    psi0 = rand_ket(N)
    t_l = LinRange(0, 200, 1000)
    fb_test1 = FloquetBasis(Ht, T)
    floquet_psi0 = to_floquet_basis(fb_test1, psi0)
    states_se = sesolve(Ht, psi0, t_l).states
    states_fse = fsesolve(fb_test1, psi0, t_l).states

    # Test overlap between from_floquet_basis and sesolve states
    for (t, state) in zip(t_l, states_se)
        from_floquet = from_floquet_basis(fb_test1, floquet_psi0, t)
        ov = abs(state' * from_floquet)
        @test isapprox(ov, 1.0; atol=2e-3)
    end

    # Test overlap between fsesolve and sesolve states
    for (state_s, state_f) in zip(states_se, states_fse)
        ov = abs(state_s' * state_f)
        @test isapprox(ov, 1.0; atol=5e-5) 
    end
end



#test Kerr-cat (working on the final version)




