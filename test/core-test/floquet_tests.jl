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


@testitem "Test Floquet Fluxonium" begin

    
    struct Fluxonium
        EL::Float64 # inductive energy
        EC::Float64 # charging energy
        EJ::Float64 # Josephson energy
        bare_levels::Int64 # bare LC levels
        Φ::Float64 # External reduced flux in units of flux quantum
    end

    function fluxonium(E_L::Float64, E_C::Float64, E_J::Float64, bare_levels::Int64, Φ::Float64)::Fluxonium
        Fluxonium(E_L, E_C, E_J, bare_levels, Φ)
    end

    function Hfl(f::Fluxonium)::QuantumObject
        ϕ = position(f.bare_levels)
        n = momentum(f.bare_levels)
        Hfl = 4f.EC * n^2 + 1 / 2 * f.EL * (ϕ + 2π * f.Φ)^2 - f.EJ * cos(ϕ)
    end

    function Hevofl(f::Fluxonium, ω::Float64)
        ϕ = position(f.bare_levels)
        g(t) = cos(ω*t)
        Hdr = 0.5*f.EL*ϕ
        return (Hfl(f), (Hdr, g))
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


# Regression test: micromotion propagators must not be cached more than once per
# in-period time, whether the duplicate arrives within one call, across calls,
# as t + kT, or through floating-point noise in mod(t + kT, T).
@testitem "Test Floquet micromotion cache has no duplicates" begin
    N = 3
    a = destroy(N)
    Ht = QobjEvo(num(N) + (a + a'), (p, t) -> cos(t))
    T = 2π
    psi0 = rand_ket(N)

    function check_cache(fb)
        @test length(fb.precompute) == length(fb.Ulist)
        @test issorted(fb.precompute)
        @test allunique(fb.precompute)
        # no two entries closer than the cache tolerance
        @test all(diff(fb.precompute) .> 1e-12)
        @test all(0 .< fb.precompute .< fb.T)
    end

    # same time repeated within a single call
    fb = FloquetBasis(Ht, T)
    propagator!(fb, [1.0, 1.0, 1.0]; progress_bar = false)
    @test length(fb.precompute) == 1
    check_cache(fb)

    # same time requested again in a later call is a cache hit
    propagator!(fb, [1.0]; progress_bar = false)
    @test length(fb.precompute) == 1

    # t and t + T map to the same in-period time
    fb = FloquetBasis(Ht, T)
    propagator!(fb, [2.0, 2.0 + T, 2.0 + 3T]; progress_bar = false)
    @test length(fb.precompute) == 1
    check_cache(fb)

    # floating-point noise: mod(0.7 + kT, T) is not bit-identical to 0.7
    fb = FloquetBasis(Ht, T)
    for k in 1:6
        propagator!(fb, [0.7 + k * T]; progress_bar = false)
    end
    @test length(fb.precompute) == 1
    check_cache(fb)

    # multiples of T (including t = 0) need no micromotion and must not be cached
    fb = FloquetBasis(Ht, T)
    propagator!(fb, [0.0, T, 2T, 5T]; progress_bar = false)
    @test isempty(fb.precompute)
    @test isempty(fb.Ulist)

    # fsesolve! on a grid spanning many periods: one entry per distinct in-period time
    fb = FloquetBasis(Ht, T)
    tl = collect(range(0, 10T, length = 41)) # 4 points per period
    fsesolve!(fb, psi0, tl, nothing, false)
    @test length(fb.precompute) == 3 # T/4, T/2, 3T/4 (t = kT needs no micromotion)
    check_cache(fb)
    fsesolve!(fb, psi0, tl, nothing, false)
    @test length(fb.precompute) == 3

    # deduplication must not change the result
    fb_ref = FloquetBasis(Ht, T)
    fb_dup = FloquetBasis(Ht, T)
    propagator!(fb_dup, [2.0, 2.0 + T]; progress_bar = false)
    @test propagator(fb_dup, 2.0).data ≈ propagator(fb_ref, 2.0).data
    @test propagator(fb_dup, 2.0 + 4T).data ≈ propagator(fb_ref, 2.0 + 4T).data
end
