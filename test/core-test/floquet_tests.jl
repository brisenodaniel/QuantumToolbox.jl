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


# Labelling of Floquet states by adiabatic connection to the eigenstates of a
# static Hamiltonian H0, through `track_quasienergies` and the
# `FloquetBasis(H0, H_dr, T)` constructor.
@testitem "Test Floquet quasienergy tracking: weak drive" begin
    using LinearAlgebra
    # columns of A and B are equal up to a phase
    phase_eq(A, B) = all(abs(abs(dot(A[:, i], B[:, i])) - 1) < 1e-6 for i in 1:size(A, 2))

    # zero drive: quasienergies are the static energies, in static order
    N = 4
    a = destroy(N)
    H0 = num(N) + 0.1 * (a + a')
    E = sort(real.(eigenenergies(H0)))
    tr = track_quasienergies(; H0 = H0, H_dr = [((p, t) -> 0.0, a + a')], T = 2π / 3.0)
    @test tr.equasi ≈ E atol = 1e-8
    @test size(tr.equasi_branches) == (N, 2)

    # weak off-resonant drive on a two-level system
    Δ, ω, A = 1.0, 3.0, 0.05
    T = 2π / ω
    H0 = 0.5Δ * sigmaz()
    H_dr = [((p, t) -> A * cos(ω * t), sigmax())]
    Ht = H0 + QobjEvo(sigmax(), (p, t) -> A * cos(ω * t))
    E = sort(real.(eigenenergies(H0)))

    # default δλ = 1 (greedy maximal overlap) agrees with a ramp
    tr = track_quasienergies(; H0 = H0, H_dr = H_dr, T = T)
    @test all(abs.(tr.equasi .- E) .< 1e-2)
    tr2 = track_quasienergies(; H0 = H0, H_dr = H_dr, T = T, δλ = 0.1, nT = 8)
    @test tr2.equasi ≈ tr.equasi atol = 1e-8
    @test length(tr2.λs) == 11

    # constructor agrees with track_quasienergies
    fb = FloquetBasis(H0, H_dr, T)
    @test fb.H isa QobjEvo
    @test fb.equasi ≈ tr.equasi atol = 1e-8
    @test phase_eq(fb.U0, tr.U0)

    # precompute and keyword arguments are forwarded
    fbp = FloquetBasis(H0, H_dr, T, [0.3T, 0.6T]; δλ = 0.5, nT = 4, tidyup_micromotion = true)
    @test fbp.precompute ≈ [0.3T, 0.6T]
    @test fbp.equasi ≈ fb.equasi atol = 1e-8
    @test phase_eq(fbp.U0, fb.U0)

    # same quasienergies (mod ω) as the default constructor
    fb_def = FloquetBasis(Ht, T)
    @test sort(mod.(fb.equasi, ω)) ≈ sort(mod.(fb_def.equasi, ω)) atol = 1e-8

    # columns of U0 are eigenvectors of U_T with eigenvalue exp(-iεT)
    UT = fb.U_T.data
    @test all(norm(UT * fb.U0[:, i] - exp(-1im * fb.equasi[i] * T) * fb.U0[:, i]) < 1e-8 for i in 1:2)

    # modes are T-periodic and fsesolve matches sesolve
    m0 = modes(fb, 0.0, Val(true))
    mT = modes(fb, T, Val(true))
    @test norm(m0 - mT) < 1e-6
    ψ0 = basis(2, 0)
    tl = LinRange(0, 5T, 23)
    s1 = sesolve(Ht, ψ0, tl; progress_bar = Val(false)).states
    s2 = fsesolve(fb, ψ0, tl, nothing, Val(false)).states
    @test all(abs(abs(x' * y) - 1) < 1e-5 for (x, y) in zip(s1, s2))
end

@testitem "Test Floquet quasienergy tracking: strong drive" begin
    ω, A = 1.0, 0.8
    T = 2π / ω
    H0 = 0.5 * sigmaz()
    H_dr = [((p, t) -> A * cos(ω * t), sigmax())]

    # a single greedy step is unreliable for a strong resonant drive
    @test_logs (:warn,) match_mode = :any track_quasienergies(; H0 = H0, H_dr = H_dr, T = T, min_overlap = 0.9)

    # a fine ramp is converged in δλ
    trA = track_quasienergies(; H0 = H0, H_dr = H_dr, T = T, δλ = 0.02, nT = 8)
    trB = track_quasienergies(; H0 = H0, H_dr = H_dr, T = T, δλ = 0.01, nT = 8)
    @test trA.equasi ≈ trB.equasi atol = 1e-6
    # unfolded branches are continuous in λ
    @test maximum(abs.(diff(trB.equasi_branches, dims = 2))) < ω / 4

    fb = FloquetBasis(H0, H_dr, T; δλ = 0.01, nT = 8)
    @test fb.equasi ≈ trB.equasi atol = 1e-6

    # original constructor still returns sorted first-zone quasienergies
    fb_old = FloquetBasis(H0 + QobjEvo(sigmax(), (p, t) -> A * cos(ω * t)), T)
    @test issorted(fb_old.equasi)
    @test all(-π / T .<= fb_old.equasi .<= π / T)

    # argument checks
    @test_throws ArgumentError track_quasienergies(; H0 = H0, H_dr = H_dr, T = T, δλ = 0.0)
    @test_throws ArgumentError track_quasienergies(; H0 = H0, H_dr = H_dr, T = T, nT = 0)
end


# Floquet calculations restricted to the subspace spanned by `basis_states`, in
# which propagators need not be unitary.
@testitem "Test Floquet subspace: modes and evolution" begin
    using LinearAlgebra
    N, k = 30, 6
    a = destroy(N)
    H0 = num(N) - 0.05 * a'^2 * a^2
    coef(p, t) = p.A * cos(p.ω * t)
    Ht = H0 + QobjEvo(a + a', coef)
    p = (A = 0.02, ω = 3.3)
    T = 2π / p.ω
    _, H0_states, _ = eigenstates(H0)
    bs = H0_states[1:k]
    V = reduce(hcat, [ψ.data for ψ in bs])

    fb_full = FloquetBasis(Ht, T; params = p, progress_bar = Val(false))
    fb = FloquetBasis(Ht, T, [0.3T]; params = p, basis_states = bs, leakage_threshold = 1e-2, progress_bar = Val(false))
    @test isnothing(fb_full.V) && fb.V ≈ V
    @test length(fb.equasi) == k && size(fb.U0) == (N, k) && size(fb.U0_dual) == (N, k)
    @test length(fb.ψfl0) == k
    @test fb.U0_dual' * fb.U0 ≈ I(k) atol = 1e-10
    @test norm(fb_full.U0_dual' * fb_full.U0 - I) < 1e-8
    # |λ| ≤ 1 up to integration error
    @test all(abs.(fb.multipliers) .<= 1 + 1e-7) && all(abs.(fb.multipliers) .> 0.99)
    @test fb.leakage_threshold == 1e-2 && fb.if_leak == :warn && fb_full.leakage_threshold == 1e-3

    # subspace modes match the corresponding full-space modes
    ω = p.ω
    for j in 1:k
        ovs = abs.(fb_full.U0' * fb.U0[:, j]) ./ norm(fb.U0[:, j])
        i = argmax(ovs)
        @test ovs[i] > 0.99
        @test abs(mod(fb.equasi[j] - fb_full.equasi[i] + ω / 2, ω) - ω / 2) < 1e-3
    end

    # fsesolve from a state in the subspace matches sesolve
    ψ0 = (bs[1] + bs[2]) / sqrt(2)
    tl = collect(range(0, 5T, length = 23))
    s1 = sesolve(Ht, ψ0, tl; params = p, progress_bar = Val(false)).states
    s2 = fsesolve(fb, ψ0, tl, nothing, Val(false)).states
    @test all(abs(x' * y) > 1 - 1e-2 for (x, y) in zip(s1, s2))
    # keyword arguments reach fsesolve, and `saveat` only selects stored states
    s3 = fsesolve(fb, ψ0, tl, nothing, Val(false); leakage_threshold = 0.5, saveat = tl[end:end]).states
    @test length(s3) == 1 && abs(s3[1]' * s1[end]) > 1 - 1e-2

    # Floquet decomposition of U_T^n, including n < 0
    Up, Um = QuantumToolbox._get_interperiod_prop(fb, [2.5T, -1.5T])
    @test norm(Up.data * Um.data - V * V') < 1e-8
    @test norm(QuantumToolbox._get_interperiod_prop(fb, [1.5T])[1].data - fb.U_T.data) < 1e-8
    # two-time propagator composes on the subspace
    U04 = propagator(fb, 0.4T)
    @test norm((propagator(fb, 0.4T, 2.7T) * U04).data - propagator(fb, 2.7T).data) < 1e-6

    # transformations to and from the Floquet basis
    t = 1.3
    F = from_floquet_basis(fb, t, Val(true))
    G = to_floquet_basis(fb, t, Val(true))
    @test size(F) == (N, k) && G * F ≈ I(k)
    @test size(from_floquet_basis(fb, t)) == (N, k)
    c = Qobj(normalize(rand(ComplexF64, k)))
    @test to_floquet_basis(fb, from_floquet_basis(fb, c, t), t).data ≈ c.data
    ρf = c * c'
    @test to_floquet_basis(fb, from_floquet_basis(fb, ρf, t), t).data ≈ ρf.data
end

@testitem "Test Floquet subspace: leakage over many periods" begin
    N = 30
    a = destroy(N)
    H0 = num(N) - 0.05 * a'^2 * a^2
    Ht = H0 + QobjEvo(a + a', (p, t) -> p.A * cos(p.ω * t))
    _, H0_states, _ = eigenstates(H0)
    bs = H0_states[1:4]
    # resonant drive, so leakage out of the subspace builds up over periods
    p = (A = 0.01, ω = 1.0)
    T = 2π / p.ω
    fb = FloquetBasis(Ht, T; params = p, basis_states = bs, if_leak = :ignore, progress_bar = Val(false))
    @test fb.if_leak == :ignore
    tlong = 40.3T
    leak = 1 - sum(abs2, QuantumToolbox._get_interperiod_prop(fb, [tlong]; if_leak = :ignore)[1].data) / 4
    @test leak > 1e-4

    # per-call options override the defaults stored in the FloquetBasis
    @test_logs propagator(fb, tlong; leakage_threshold = leak / 2) # stored if_leak = :ignore
    @test_logs (:warn, r"over 40 periods") propagator(fb, tlong; leakage_threshold = leak / 2, if_leak = :warn)
    @test_logs propagator(fb, tlong; leakage_threshold = 2leak, if_leak = :warn)
    @test_throws ErrorException propagator(fb, tlong; leakage_threshold = leak / 2, if_leak = :error)
    @test_throws ErrorException fsesolve(
        fb, bs[1], [0.0, tlong], nothing, Val(false); leakage_threshold = leak / 2, if_leak = :error,
    )

    # stored defaults apply when no per-call options are given
    fb2 = FloquetBasis(Ht, T; params = p, basis_states = bs, leakage_threshold = leak / 2, if_leak = :error, progress_bar = Val(false))
    @test_throws ErrorException propagator(fb2, tlong)
    @test_logs propagator(fb2, tlong; leakage_threshold = 2leak)
end

@testitem "Test Floquet subspace: tracking, composite systems and argument checks" begin
    using LinearAlgebra
    N, k = 30, 6
    a = destroy(N)
    H0 = num(N) - 0.05 * a'^2 * a^2
    p = (A = 0.02, ω = 3.3)
    T = 2π / p.ω
    Ht = H0 + QobjEvo(a + a', (p, t) -> p.A * cos(p.ω * t))
    _, H0_states, _ = eigenstates(H0)
    bs = H0_states[1:k]

    # tracking constructor in a subspace agrees with the full-space labelling
    H_dr = [((q, t) -> 0.02 * cos(3.3t), a + a')]
    ft = FloquetBasis(H0, H_dr, T; n_basis_states = k, leakage_threshold = 1e-2)
    ft_full = FloquetBasis(H0, H_dr, T)
    @test length(ft.equasi) == k
    @test ft.equasi ≈ ft_full.equasi[1:k] atol = 1e-3
    @test all(diag(abs.(ft_full.U0[:, 1:k]' * ft.U0)) .> 0.99)

    # composite system keeps its dimensions
    b = destroy(3)
    c = destroy(4)
    Hc0 = tensor(num(3), qeye(4)) + 1.3 * tensor(qeye(3), num(4)) + 0.1 * tensor(b + b', c + c')
    Hc = Hc0 + QobjEvo(tensor(b + b', qeye(4)), (q, t) -> 0.01 * cos(3.1t))
    _, Hc_states, _ = eigenstates(Hc0)
    fbc = FloquetBasis(Hc, 2π / 3.1; basis_states = Hc_states[1:5], progress_bar = Val(false))
    @test fbc.U_T.dims == [3, 4] && propagator(fbc, 1.1).dims == [3, 4]
    @test size(from_floquet_basis(fbc, 0.7)) == (12, 5)
    @test modes(fbc, 0.7)[1].dims == [3, 4]

    # argument checks
    @test_throws ArgumentError FloquetBasis(Ht, T; params = p, basis_states = bs, integrate_columns = false)
    @test_throws ArgumentError FloquetBasis(Ht, T; params = p, basis_states = [basis(5, 0)])
    @test_throws ArgumentError FloquetBasis(Ht, T; params = p, if_leak = :foo)
    fb = FloquetBasis(Ht, T; params = p, basis_states = bs, leakage_threshold = 1e-2, progress_bar = Val(false))
    @test_throws ArgumentError propagator(fb, 0.77T; integrate_columns = false)
    @test_throws ArgumentError propagator(fb, 0.77T; if_leak = :foo)
    @test_throws ArgumentError FloquetBasis(H0, H_dr, T; n_basis_states = k, basis_states = bs)
end
