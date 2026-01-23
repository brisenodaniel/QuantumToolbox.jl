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



#test Kerr-cat (working in progress)
@testitem "Test Floquet Kerr cat" begin
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

    eigs = zeros(Float64, length(Ω_d), N)
    for (idx,Omd) in enumerate(Ω_d)
        Π = 4 * Omd / (3 * ωd)
        Δ = ω0 - ωd/2 + 6 * g4 * Π^2 - 18 * g3^2 * Π^2 / ωd + 2 * K
        ϵ2 = g3 * Π
        Ham_K = Δ * a_d * a - (K / 2) * a_d * a_d * a * a + ϵ2 * (a_d * a_d + a * a)
        eigs[idx, :] .= real.(eigenenergies(Ham_K))
    end
end


