export FloquetBasis, FloquetEvolutionSol, track_quasienergies, propagator, propagator!, fsesolve, fsesolve!, to_floquet_basis, to_floquet_basis!, from_floquet_basis, from_floquet_basis!, modes, modes!, states, states!
# script helper functions
function _to_period_interval(tlist::AbstractVector, T::Real)
    # function maps all elements ``t`` in `tlist` outside the interval ``[0, T)`` to an equivalent
    # time ``\tau`` such that ``mod(t, T) = \tau``
    if !isempty(tlist)
        tlist = mod.(tlist, T) |> unique |> sort
    end
    return tlist
end

function quasienergy_sort(
    U_T::AbstractQuantumObject,
    T::Real;
    track::Union{Nothing, NamedTuple}=nothing,
    )
    # uns for unsorted
    ens, ψfl0_uns, U0_uns = eigenstates(U_T)
    ε_uns = -angle.(ens) ./ T
    if isnothing(track)
        # sort by quasienergy in the first Floquet zone
        P = sortperm(ε_uns)
        ε = ε_uns[P]
    else
        # match to the states returned by `track_quasienergies`, and unfold the
        # quasienergies onto the tracked branches
        P = _greedy_assignment(abs2.(track.U0' * U0_uns))
        ε = _unfold_quasienergies(ε_uns[P], track.equasi, T)
    end
    ψfl0 = ψfl0_uns[P]
    U0 = U0_uns[:, P]
    return ε, ψfl0, U0
end

function _greedy_assignment(O::AbstractMatrix{<:Real})
    # returns `P` such that `P[i]` is the column of `O` assigned to row `i`,
    # repeatedly assigning the largest remaining entry of `O`
    O = copy(O)
    N = size(O, 1)
    P = zeros(Int, N)
    for _ in 1:N
        i, j = Tuple(argmax(O))
        P[i] = j
        O[i, :] .= -Inf
        O[:, j] .= -Inf
    end
    return P
end

function _unfold_quasienergies(ε::AbstractVector, ε_ref::AbstractVector, T::Real)
    # shift each quasienergy by a multiple of ω = 2π/T to the value closest to `ε_ref`
    ω = 2π / T
    return ε .+ ω .* round.((ε_ref .- ε) ./ ω)
end

@doc raw"""
    track_quasienergies(; H0, H_dr, T, nT=1, δλ=1.0, params=nothing, min_overlap=0.5, alg=Vern7(lazy=false), reltol=1e-12, kwargs...)

Label the Floquet states of ``\hat{H}(t) = \hat{H}_0 + \sum_k f_k(p, t) \hat{H}_k`` by the eigenstate of
``\hat{H}_0`` that they are adiabatically connected to.

The drive is ramped as ``\hat{H}_\lambda(t) = \hat{H}_0 + \lambda \sum_k f_k(p, t) \hat{H}_k`` for
``\lambda = 0, \delta\lambda, \dots, 1``. At each step the Floquet states are matched to those of the
previous step by maximal period-averaged overlap, and the quasienergies are unfolded so that each branch
is continuous in ``\lambda``, starting from the eigenvalues of ``\hat{H}_0``.

# Arguments:
- `H0`: Static Hamiltonian.
- `H_dr`: Vector of drive terms `(f_k, H_k)`, with coefficient functions `f_k(p, t)`.
- `T`: Drive period.
- `nT`: Number of equally spaced times in ``[0, T)`` over which overlaps are averaged. The default `nT=1` compares states at ``t=0`` only.
- `δλ`: Step in the drive-strength ramp. The default `δλ=1` matches the Floquet states directly to the static eigenstates by maximal overlap, which is sufficient for weak drives.
- `params`: Parameters passed to the coefficient functions `f_k`.
- `min_overlap`: A warning is raised if any matched overlap in a step is below this value, which indicates that `δλ` should be decreased.
- `alg`, `reltol`, `kwargs`: Passed to [`sesolve`](@ref) when computing the period propagators.

# Returns:
A `NamedTuple` with fields
- `equasi`: Unfolded quasienergies at ``\lambda = 1``, where `equasi[i]` is connected to the `i`-th eigenvalue of `H0` (ascending order).
- `U0`: Matrix whose `i`-th column is the corresponding Floquet state at ``t = 0``.
- `λs`: The values of ``\lambda`` used.
- `equasi_branches`: Matrix whose `(i, k)` entry is the `i`-th quasienergy at `λs[k]`.
"""
function track_quasienergies(;
                             H0::QuantumObject,
                             H_dr::AbstractVector{<:Tuple{Function, AbstractQuantumObject}},
                             T::Real,
                             nT::Int=1,
                             δλ::Real=1.0,
                             params::PP=nothing,
                             min_overlap::Real=0.5,
                             alg::AbstractODEAlgorithm=Vern7(lazy=false),
                             reltol::Real=1e-12,
                             kwargs...
                             ) where {PP}
    T > 0 || throw(ArgumentError("`T` must be a nonzero positive real number"))
    nT >= 1 || throw(ArgumentError("`nT` must be a positive integer"))
    0 < δλ <= 1 || throw(ArgumentError("`δλ` must lie in the interval (0, 1]"))
    isempty(H_dr) && throw(ArgumentError("`H_dr` must contain at least one drive term"))
    # always end the ramp at λ = 1
    λs = range(0.0, 1.0, length=ceil(Int, 1 / δλ - sqrt(eps())) + 1)
    # sample nT times in [0, T), plus T for the period propagator
    tlist = range(0.0, Float64(T), length=nT + 1)
    # λ = 0: use the eigenstates of H0 directly, since U(T) is degenerate for
    # static energies that differ by a multiple of 2π/T
    E, _, V0 = eigenstates(H0)
    E = real.(E)
    P = sortperm(E)
    εref = E[P]
    Ψref = V0[:, P]
    # static eigenstates only acquire a phase in time
    Ψref_t = fill(Ψref, nT)
    ε_branches = Matrix{Float64}(undef, length(εref), length(λs))
    ε_branches[:, 1] = εref

    for (k, λ) in enumerate(λs[2:end])
        H_λ = QobjEvo(Tuple((H_int, (p, t) -> λ * dr(p, t)) for (dr, H_int) in H_dr))
        Us = sesolve(
            H0 + H_λ,
            qeye_like(H0),
            tlist;
            params=params,
            alg=alg,
            reltol=reltol,
            progress_bar=Val(false),
            kwargs...
        ).states
        vals, _, V = eigenstates(Us[end])
        ε = -angle.(vals) ./ T
        # Floquet states at the sampled times
        Ψ_t = [U.data * V for U in Us[1:nT]]
        # period-averaged overlap O[i, j] with the states of the previous step
        O = sum(abs2.(Ψr' * Ψ) for (Ψr, Ψ) in zip(Ψref_t, Ψ_t)) ./ nT
        P = _greedy_assignment(O)
        worst = minimum(O[i, P[i]] for i in eachindex(P))
        if worst < min_overlap
            @warn "Minimum matched overlap $(round(worst, digits=3)) at λ = $λ is below " *
                "`min_overlap = $min_overlap`. The labelling may be unreliable; consider decreasing `δλ`."
        end
        εref = _unfold_quasienergies(ε[P], εref, T)
        Ψref = V[:, P]
        Ψref_t = [Ψ[:, P] for Ψ in Ψ_t]
        ε_branches[:, k + 1] = εref
    end
    return (; equasi=εref, U0=Ψref, λs=collect(λs), equasi_branches=ε_branches)
end

function qeye_like(A::QobjEvo, ::Val{true})
    basis_dim = size(A)[1]
    return qeye(basis_dim, dims=A.dims)
end

function qeye_like(A::QobjEvo, ::Val{false})
    return qeye_like(A)
end

function qeye_like(A::QuantumObject, ::Val{T}) where {T}
    return qeye_like(A)
end



function _data_to_ketlist(M::AbstractMatrix, dims::DT) where {DT<:AbstractVector{Int}}
    M_list = eachcol(M) |> collect
    ψ_list = Qobj.(M_list, dims=dims)
    return ψ_list
end

function _state_mtrx_to_mode(
    M::AbstractMatrix,
    equasi::AbstractVector{Float64},
    t::Float64)
    ϕ_mat = exp.(1im * t .* equasi) |> Diagonal
    return M * ϕ_mat
end

struct FloquetEvolutionSol{
    TT1<:AbstractVector,
    TT2<:AbstractVector,
    TQ<: AbstractQuantumObject,
    TS<:AbstractVector{TQ},
    TE,
    AlgT<:AbstractODEAlgorithm,
    TolT<:Real
}
    times::TT1
    times_states::TT2
    states::TS
    expect::AbstractArray{TE}
    alg::AlgT
    abstol_UT::TolT
    reltol_UT::TolT
    abstol::TolT
    reltol::TolT
end



@doc raw"""
   struct FloquetBasis

Julia struct containing propagators, quasienergies, and Floquet states for a system with a ``T``-periodic Hamiltonain.

# Fields:
- `H::AbstractQuantumObject`: T-periodic Hamiltonian.
- `T<:Real`: Hamiltonian period such that ``\hat{H}(T) = \hat{H}(0)``
- `tlist::TT`: Time array fed to `sesolve` to compute propagators and Floquet states. First and final elements are always `0, T`. All elements lie in range `[0,T]`, see notes for behavior when field is set to an array with points outside this range.
- `precompute::TT`: Times for which the micromotion propagator and Floquet modes are precomputed. When initializing this struct, this field may be left blank, set to a Bool, or set to a list of timepoints. If set to `false`, no propagators will be stored except for the final period Hamiltonian. If left blank or set to `true`, this field will be set to the same value as `tlist`.  All elements  lie in range `[0,T]`. See notes for behavior when field is set to an array with points outside this range.
- `U_T::Qobj`: System propagator at time ``T``
- `Ulist::AbstractVector{Qobj}`: List of micromotion propagators at times `tlist`
- `equasi::TE`: Time-independent quasienergies
- `params::AbstractVector` : Additional parameters for the time-dependent Hamiltonian to be used in sesolve.
"""
struct FloquetBasis{
    TQ<:AbstractQuantumObject,
    TS<:AbstractQuantumObject,
    TA<:AbstractODEAlgorithm,
    TZ<:Complex,
    PP,
}
    H::Union{Qobj, QobjEvo}
    T::Float64
    precompute::Vector{Float64}
    U_T::Qobj
    ψfl0::Vector{TS}
    U0::Matrix{TZ}
    Ulist::Vector{TQ}
    equasi::Vector{Float64}
    alg::TA
    abstol_UT::Float64
    reltol_UT::Float64
    params::PP
    kwargs::NamedTuple

    @doc raw"""
        FloquetBasis(H::AbstractQuantumObject, T::Real, tlist::AbstractVector{Real}, precompute::Bool = true; kwargs::Dict = Dict())

        DOCSTRING

        # Arguments:
        - `H`: Time-dependent system Hamiltonian.
        - `T`: Hamiltonian period such that ``\hat{H}(T+\tau_0) = \hat{H}(\tau_0)``.
        - `precompute`: Time vector containing points ``t`` at which to store the system propagator ``U(t)``.
        - `kwargs`: Additional keyword arguments to pass to ssesolve.

        # Notes:
        - If `precompute` contains elements outside the interval ``[0,T]``, a new time vector will be produced with all times ``t_k`` not in the interval mapped to an equivalent time ``\tau_k`` in the interval such that ``\hat{H}(t_k) = \hat{H}(\tau_k)``.

        # Returns:
        - `fbasis::FloquetBasis`: FloquetBasis object for the system evolving under the ``T``-periodic Hamiltonian `H`.
        """
    function FloquetBasis(
        H::AbstractQuantumObject,
        T::Real,
        precompute::Union{AbstractVector, Nothing}=nothing;
        params::PP=nothing,
        alg::AbstractODEAlgorithm = Vern7(lazy = false),
        reltol_UT::Real=1e-12, # smaller tolerance for error when computing period-prop
        abstol_UT::Union{Real, Nothing}=nothing, # keep error floor at default unless user specifies otherwise
        tidyup_UT::Bool=true, # set noise floor of period-propagator at abstol of solver
        tidyup_micromotion::Bool=false, # set noise floor of micromotion operators at abstol of solver
        kwargs...
        ) where {PP}
        kwargs = (; kwargs...)
        if T<=0
            throw(
                ArgumentError("`T` must be a nonzero positive real number")
            )
        end
        precompute = isnothing(precompute) ? Float64[] : Float64.(precompute)
        T = Float64(T)
        # enforce `precompute` interval rule
        precompute = _to_period_interval(precompute, T)
        # remove 0 from precompute if present, since these timesteps have no micromotion
        if !isempty(precompute) && precompute[1] == 0.0
            popfirst!(precompute)
        end
        tlist = Float64[0, precompute..., T]
        # assemble integrator kwargs used for propagator
        kwargs_UT = (kwargs..., saveat=tlist, reltol=reltol_UT)
        if !isnothing(abstol_UT)
            kwargs_UT = (kwargs_UT..., abstol=abstol_UT)
        end
        # compute propagators
        sol = sesolve(
            H,
            qeye_like(H, Val(true)),
            tlist;
            params=params,
            alg=alg,
            kwargs_UT...
                )
        Ulist = sol.states[2:end-1] # don't put t=0 or t=T in micromotion cache
        U_T = sol.states[end]
        # tidyup period-propagator and micromotion propagators according to boolean flags
        if tidyup_UT
            tidyup!(U_T, sol.abstol)
        end
        if tidyup_micromotion
            tidyup!.(Ulist, sol.abstol)
        end
        # solve for quasienergies
        equasi, ψfl0, U0 = quasienergy_sort(U_T, T)
        TQ, TS, TA, TZ = eltype(Ulist), eltype(ψfl0), typeof(sol.alg), eltype(U0)

        new{TQ, TS, TA, TZ, PP}(
            H,
            Float64(T),
            precompute,
            U_T,
            ψfl0,
            U0,
            Ulist,
            equasi,
            sol.alg,
            sol.abstol,
            sol.reltol,
            params,
            kwargs,
        )
    end

    @doc raw"""
        FloquetBasis(H0::QuantumObject, H_dr::AbstractVector{<:Tuple{Function, AbstractQuantumObject}}, T::Real, precompute::Union{AbstractVector, Nothing}=nothing; δλ::Real=1.0, nT::Int=1, min_overlap::Real=0.5, kwargs...)

    FloquetBasis for the ``T``-periodic Hamiltonian ``\hat{H}(t) = \hat{H}_0 + \sum_k f_k(p, t) \hat{H}_k``, with
    quasienergies and Floquet states labelled by the eigenstate of ``\hat{H}_0`` they are adiabatically connected to.

    # Arguments:
    - `H0`: Static Hamiltonian.
    - `H_dr`: Vector of drive terms `(f_k, H_k)`, with coefficient functions `f_k(p, t)`.
    - `T`: Drive period.
    - `precompute`: Time vector containing points ``t`` at which to store the system propagator ``U(t)``.
    - `δλ`, `nT`, `min_overlap`: Options for the drive-strength ramp, see [`track_quasienergies`](@ref). The default `δλ=1` assigns each Floquet state to the static eigenstate with which it has maximal overlap, which is sufficient for weak drives.
    - `kwargs`: Remaining keyword arguments are the same as for `FloquetBasis(H, T, precompute)`, and are also used when computing the propagators of the ramp.

    # Returns:
    - `fbasis::FloquetBasis`: FloquetBasis object whose `i`-th quasienergy and Floquet state are connected to the `i`-th eigenstate of `H0` (ascending energy). Quasienergies are unfolded onto these branches rather than restricted to ``[-\pi/T, \pi/T)``.
    """
    function FloquetBasis(
        H0::QuantumObject,
        H_dr::AbstractVector{<:Tuple{Function, AbstractQuantumObject}},
        T::Real,
        precompute::Union{AbstractVector, Nothing}=nothing;
        δλ::Real=1.0,
        nT::Int=1,
        min_overlap::Real=0.5,
        params::PP=nothing,
        alg::AbstractODEAlgorithm = Vern7(lazy = false),
        reltol_UT::Real=1e-12,
        abstol_UT::Union{Real, Nothing}=nothing,
        kwargs...
        ) where {PP}
        H = H0 + QobjEvo(Tuple((H_k, f_k) for (f_k, H_k) in H_dr))
        fb = FloquetBasis(
            H,
            T,
            precompute;
            params=params,
            alg=alg,
            reltol_UT=reltol_UT,
            abstol_UT=abstol_UT,
            kwargs...
        )
        # ramp the drive with the same solver settings used for the period propagator
        kwargs_track = (; (k => v for (k, v) in pairs(kwargs) if k ∉ (:tidyup_UT, :tidyup_micromotion))...)
        if !isnothing(abstol_UT)
            kwargs_track = (kwargs_track..., abstol=abstol_UT)
        end
        track = track_quasienergies(;
            H0=H0,
            H_dr=H_dr,
            T=T,
            nT=nT,
            δλ=δλ,
            params=params,
            min_overlap=min_overlap,
            alg=alg,
            reltol=reltol_UT,
            kwargs_track...
        )
        # relabel the Floquet states of `fb` by the tracked branches
        equasi, ψfl0, U0 = quasienergy_sort(fb.U_T, fb.T; track=track)
        TQ, TS, TA, TZ = eltype(fb.Ulist), eltype(ψfl0), typeof(fb.alg), eltype(U0)

        new{TQ, TS, TA, TZ, PP}(
            fb.H,
            fb.T,
            fb.precompute,
            fb.U_T,
            ψfl0,
            U0,
            fb.Ulist,
            equasi,
            fb.alg,
            fb.abstol_UT,
            fb.reltol_UT,
            params,
            fb.kwargs,
        )
    end
end


function Base.getproperty(fb::FloquetBasis, key::Symbol)
    if key==:dims
        return fb.H.dims
    else
        return Core.getfield(fb, key)
    end
end

function qeye_like(fb::FloquetBasis)
    return qeye_like(fb.U_T)
end


function _init_FloquetEvolutionSol(
    fb::FloquetBasis,
    tlist::AbstractVector,
    e_ops::Union{Nothing, AbstractVector, Tuple} = nothing;
    kwargs...
)
    # create editable version of kwargs
    sol_kwargs = Dict{Symbol, Any}(kwargs)
    # determine if expectation values need to be calculated
    has_eops = !(isnothing(e_ops) || isempty(e_ops))
    # determine timesteps at which to store propagated state
    nsteps = length(tlist)
    if haskey(sol_kwargs, :saveat)
        # entry not checked by multiple dispatch, check if of correct type
        if !(sol_kwargs[:saveat] isa AbstractVector{<:Real})
            throw(
                TypeError(:fsesolve, "saveat", AbstractVector{<:Real}, typeof(sol_kwargs[:saveat]))
            )
        end
    elseif !has_eops
        sol_kwargs[:saveat] = tlist
    else
        sol_kwargs[:saveat] = [tlist[end]]
    end
    nstates = length(sol_kwargs[:saveat])

    # determine whether expectation values must be collected
    #  if so, determine size
    n_eops = has_eops ? length(e_ops) : 0
    # pre-allocate solution memory

    sol = FloquetEvolutionSol(
        tlist,
        sol_kwargs[:saveat],
        Vector{QuantumObject{Ket}}(undef, nstates),
        has_eops ? Array{ComplexF64}(undef, nsteps, n_eops) : ComplexF64[],
        fb.alg,
        fb.abstol_UT,
        fb.reltol_UT,
        haskey(sol_kwargs, :abstol) ? sol_kwargs[:abstol] : 0.0,
        haskey(sol_kwargs, :reltol) ? sol_kwargs[:reltol] : 0.0,
    )
    return sol
end

@doc raw"""
    FloquetBasis(H::AbstractQuantumObject, T::Real; kwargs::Dict = Dict())

DOCSTRING

# Arguments:
- `H`: Time-dependent system Hamiltonian.
- `T`: Hamiltonian period such that ``\hat{H}(T+\tau_0) = \hat{H}(\tau_0)``.
- `kwargs`: Additional keyword arguments to pass to ssesolve.

# Notes
- Calling `FloquetBasis` without providing a time vector will create a FloquetBasis object with default `tlist=range(start:0, stop:T, length:101)`.

# Returns:
- `fbasis::FloquetBasis`: Floquet basis object for the system evolving under the time-dependent Hamiltonian `H`.
"""

function memoize_micromotion!(
    fb::FloquetBasis,
    tlist::AbstractVector;
    kwargs...
    )
    tlist = _to_period_interval(tlist, fb.T)
    propagator!(fb, tlist; kwargs...)
end

function memoize_micromotion!(fb::FloquetBasis, t::Float64, U::QuantumObject)
    t = mod(t, fb.T)
    # times equivalent to t=0 have no micromotion, and times already cached
    # (up to `_MEMOIZE_TOL`) must not be stored twice
    if t <= _MEMOIZE_TOL || !isnothing(_memoized_index(fb, t))
        return nothing
    end
    t_idx = findfirst(x -> x>t, fb.precompute)
    t_idx = isnothing(t_idx) ? length(fb.precompute) + 1 : t_idx
    insert!(fb.precompute, t_idx, t)
    insert!(fb.Ulist, t_idx, U)
end

function memoize_micromotion!(
    fb::FloquetBasis,
    tlist::AbstractVector,
    Ulist::AbstractVector{<:AbstractQuantumObject}
    )
    for (t, U) in zip(tlist, Ulist)
        memoize_micromotion!(fb, t, U)
    end
end


######## Propagator refactor


# Absolute tolerance on in-period times. `mod(t + kT, T)` is generally not
# bit-identical to `t`, so cache lookups must be tolerance-based.
const _MEMOIZE_TOL = 1e-12

function _memoized_index(fb::FloquetBasis, t::Real)
    # `t` must already lie in [0, T); `fb.precompute` is kept sorted
    idx = searchsortedfirst(fb.precompute, t - _MEMOIZE_TOL)
    if idx <= length(fb.precompute) && fb.precompute[idx] <= t + _MEMOIZE_TOL
        return idx
    end
    return nothing
end

function is_memoized(fb::FloquetBasis, t::Real)
    t = mod(t, fb.T)
    return t <= _MEMOIZE_TOL || !isnothing(_memoized_index(fb, t))
end

function _from_micromotion_cache(fb::FloquetBasis, t::Real)
    if is_memoized(fb, t)
        t = mod(t, fb.T)
        return t <= _MEMOIZE_TOL ?
            qeye_like(fb) : fb.Ulist[_memoized_index(fb, t)]
    else
        return nothing
    end
end

function _compute_micromotion(
    fb::FloquetBasis,
    tlist::AbstractVector,
    pbar::Bool;
    kwargs...
    )
    # send tlist to period interval
    # NOTE: order of timesteps may change if any timestep is outside the
    # interval [0,T)
    trem_list = _to_period_interval(tlist, fb.T)
    # ensure that first timestep is zero for proper convergence
    t0 = trem_list[1]
    solver_t = t0 == zero(t0) ? trem_list : [zero(t0), trem_list...]
    # calculate micromotion propagators
    U_micros = sesolve(
        fb.H,
        qeye_like(fb),
        solver_t;
        alg=fb.alg,
        progress_bar=Val(pbar),
        kwargs...
    ).states
    # remove zero timestep if it was added
    if t0 != zero(t0)
        popfirst!(U_micros)
    end
    # micromotion list is in order of t_rem
    # send to tlist ordering
    U_micros = U_micros[indexin(mod.(tlist,fb.T), trem_list)]
    return U_micros
end

function _get_micromotion(
    fb::FloquetBasis,
    tlist::AbstractVector,
    pbar::Bool;
    precomputed_only::Bool=false,
    kwargs...
    )
    # initialize list of micromotion propagators
    Ulist = Vector{QuantumObject}(undef, length(tlist))
    # get memoized propagators
    Umem = [_from_micromotion_cache(fb, t) for t in tlist]
    # get uncached timesteps
    isnew = isnothing.(Umem)
    tnew = tlist[isnew]
    # check to see if new micromotion is needed and allowed
    if precomputed_only && !isempty(tnew)
        throw(
            ErrorException(
                "Keyword argument `precomputed_only` set to true, "*
                    "but micromotion propagators for timesteps $tnew "*
                    "are not precomputed in FloquetBasis argument"
            )
        )

    end
    # remove `nothing` elements from Umem
    Umem = all(isnew) ? [] : Umem[.~isnew]
    # compute new micromotion
    Unew = any(isnew) ?  _compute_micromotion(fb, tnew, pbar; kwargs...) : []
    # populate return array
    for (Us, idxs) in zip([Unew, Umem], [isnew, .~isnew])
        if any(idxs)
            Ulist[idxs] = Us
        end
    end
    return Ulist, Unew, tnew
end

function _get_interperiod_prop(
    fb::FloquetBasis,
    tlist::AbstractVector
    )
    # get period counts for each timestep
    nT_list = fld.(tlist, fb.T)
    # avoid repeating matrix exponentiations
    nT_unq = unique(nT_list)
    # propagate over periods
    UnT = map(nT -> fb.U_T^nT, nT_unq)
    # re-add repeated propagators if needed
    UnT = nT_list.size == nT_unq.size ? UnT : UnT[indexin(nT_list, nT_unq)]
    return UnT
end

function _proplist(
    fb::FloquetBasis,
    tlist::AbstractVector,
    pbar::Bool;# TODO: Change to Val consistent with rest of QT.jl
    kwargs...
    )
    pbar ? println("Calcuating Micromotion Propagators") : nothing
    Umicro_list, Unew_list, tnew = _get_micromotion(
        fb,
        tlist,
        pbar;
        kwargs...
    )
    pbar ? println("Finished.") : nothing
    UnT_list = _get_interperiod_prop(fb, tlist)
    # compute total propagators
    Utot = [Umicro * UnT for (Umicro, UnT) in zip(Umicro_list, UnT_list)]
    return Utot, UnT_list, Umicro_list, Unew_list, tnew
end


function propagator(
    fb::FloquetBasis,
    tlist::AbstractVector;
    progress_bar::Bool=false,
    kwargs...)
    return _proplist(fb, tlist, progress_bar; kwargs...)[1]
end

function propagator!(
    fb::FloquetBasis,
    tlist::AbstractVector;
    progress_bar::Bool=false,
    kwargs...
    )
    Ulist, _, _, Unew, tnew = _proplist(fb, tlist, progress_bar; kwargs...)
    if !isempty(tnew)
        memoize_micromotion!(fb, tnew, Unew)
    end
    return Ulist
end


function propagator(fb::FloquetBasis, t::Real; kwargs...)
    return propagator(fb, [t]; kwargs...)[end]
end

function propagator!(fb::FloquetBasis, t::Real; kwargs...)
    return propagator!(fb, [t]; kwargs...)[end]
end

function propagator(fb::FloquetBasis, t0::Real, tf::Real; kwargs...)
    Ulist = propagator(fb, [t0, tf]; kwargs...)
    if t0==zero(t0)
        return Ulist[end]
    else
        # Ulist contains propagators [U(0, t0), U(0, tf)]
        # Return U(t0, tf) = U(0, tf) * U'(0, t0)
        return Ulist[end] * Ulist[1]'
    end
end

function propagator!(fb::FloquetBasis, t0::Real, tf::Real; kwargs...)
    Ulist = propagator!(fb, [t0, tf]; kwargs...)
    if t0==zero(t0)
        return Ulist[end]
    else
        # Ulist contains propagators [U(0, t0), U(0, tf)]
        # Return U(t0, tf) = U(0, tf) * U'(0, t0)
        return Ulist[end] * Ulist[1]'
    end
end
#
#function propagator(fb::FloquetBasis, t0::TI, tf::TF; kwargs...) where {TI<:Real, TF<:Real}
#t0, tf = Float64[t0, tf] # ensure timepoints are Float64
#U0, U_nT, U_intra = _prop_list(fb, t0, tf; kwargs...)
#return U_intra * U_nT * U0'
#end
#
## TODO: Current implementation of Propagator makes fsesolve slower than sesolve.
## This can be fixed by forcing fesolve to make only one call to sesolve, in which
## all required uncached micromotion operators are calculated in that single call.
#
#function propagator!(fb::FloquetBasis, t0::TI, tf::TF; kwargs) where{TI<:Real, TF<:Real}
#t0, tf = Float64[t0, tf] # ensure timepoints are Float64
#U0, U_nT, U_intra = _prop_list(fb, t0, tf; kwargs...)
#t0_T, tf_T = mod.([t0, tf], fb.T)
#for (tp, U) in [(t0_T, U0), (tf_T, U_intra)]
#if !(tp==0 || tp∈fb.precompute)
#memoize_micromotion!(fb, tp, U)
#end
#end
#return U_intra * U_nT * U0'
#end
#
#function _prop_list(fb::FloquetBasis, t0::Float64, tf::Float64; kwargs...)
#U0 = (t0 == 0.0) ? qeye_like(fb.H, Val(true)) : propagator(fb, 0.0, t0; kwargs...)
#nT, t_rem = fldmod(tf, fb.T)
#U_nT = fb.U_T^nT
#if t_rem == 0.0
#U_intra = qeye_like(fb.H, Val(true))
#elseif t_rem∈fb.precompute
#t_idx = findfirst(x->x==t_rem, fb.precompute)
#U_intra = fb.Ulist[t_idx]
#else
#if haskey(kwargs, :memoized_only) && kwargs[:memoized_only]
#throw(
#ErrorException(
#"`memoized_only` keyword argument set to `true`, but "*
#"the micromotion propagatoo `tf=$tf` "*
#"is not memoized in fb."
#)
#)
#end
#tlist = Float64[0.0, t_rem]
#kwargs = Dict(fb.kwargs..., kwargs...)
#U_intra = sesolve(
#fb.H,
#qeye_like(fb.H, Val(true)),
#tlist;
#alg=fb.alg,
#progress_bar=Val(false),
#kwargs...
#).states[end]
#end
#return U0, U_nT, U_intra
#end

"""
    fsesolve(
    fb::FloquetBasis,
    ψ0::QuantumObject{Ket},
    tlist::AbstractVector,
    alg::AbstractODEAlgorithm = Vern7(lazy=false),
    e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
    params=NullParameters(),
    progress_bar::Union{Val, Bool} = Val(true),
    inplace::Union{Val,Bool}=Val(true),
    kwargs...;
    exact_t::Bool = false,
)

TBW
"""
function fsesolve(
    fb::FloquetBasis,
    ψ0::QuantumObject{Ket},
    tlist::AbstractVector{TS},
    e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
    progress_bar::Union{Val, Bool} = Val(true),
    kwargs...,
) where {TS<:Real}
    return _fsesolve(fb, ψ0, tlist, propagator, e_ops, progress_bar; kwargs...)
end


function fsesolve!(
    fb::FloquetBasis,
    ψ0::QuantumObject{Ket},
    tlist::AbstractVector{TS},
    e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
    progress_bar::Union{Val, Bool} = Val(true),
    kwargs...,
) where {TS<:Real}
    return _fsesolve(fb, ψ0, tlist, propagator!, e_ops, progress_bar; kwargs...)
end


function _fsesolve(
    fb::FloquetBasis,
    ψ0::QuantumObject{Ket},
    tlist::AbstractVector{TS},
    pfunc::Function,
    e_ops::Union{AbstractVector, Tuple, Nothing} = nothing,
    progress_bar::Union{Val, Bool} = Val(true);
    kwargs...,
) where {TS<:Real}
    sol = _init_FloquetEvolutionSol(
        fb,
        tlist,
        e_ops,
        kwargs...
    )
    # convert progress_bar to boolean type
    if !(progress_bar isa Bool)
        progress_bar = typeof(progress_bar).parameters[1]
    end
    # get propagators
    Ulist = pfunc(fb, tlist; progress_bar=progress_bar, kwargs...)
    # get indices of timesteps at which to store state
    state_idxs = indexin(tlist, sol.times_states)

    # collect expval at each t (if e_ops not Nothing) and state at every :saveat
    for (tstep, U) in enumerate(Ulist)
        # propagate state
        ψt = U * ψ0
        # get expvals
        if !isempty(sol.expect)
            sol.expect[tstep, :] = [expect(eop, ψt) for eop in e_ops]
        end
        # save state if timestep in :saveat
        ψ_idx = state_idxs[tstep]
        if !isnothing(ψ_idx)
            sol.states[ψ_idx] = ψt
        end
    end
    return sol
end


###### State and Mode helper functions

######## Get  modes at time 0

function modes(fb::FloquetBasis, ::Val{true}; kwargs...)
    return fb.U0
end

function modes(fb::FloquetBasis, ::Val{false}=Val(false); kwargs...)
    return modes(fb, Val(true)) |> x -> _data_to_ketlist(x, fb.dims)
end

# at t=0, state function is alias for modes function, since the Floquet
# state and modes conicide at t=0
function states(fb::FloquetBasis, ::Val{true}; kwargs...)
    return modes(fb, Val(true))
end

function states(fb::FloquetBasis, ::Val{false}=Val(false); kwargs...)
    return modes(fb, Val(false))
end


function _states(
    fb::FloquetBasis,
    tlist::AbstractVector,
    pfunc::Function;
    kwargs...
    )
    U0 = modes(fb, Val(true))
    Uts = [U.data for U in pfunc(fb, tlist; kwargs...)]
    ψts = Uts .* Ref(U0)
    tlist[1] == zero(tlist[1]) ? ψts[1] = U0 : nothing
    return ψts
end

function _states(
    fb::FloquetBasis,
    t::Real,
    pfunc::Function;
    kwargs...
    )
    return _states(fb, [t], pfunc; kwargs...)[1]
end


#function _states(fb::FloquetBasis, t::Real, pfunc::Function; kwargs...)
    ## This function is defined to avoid code-repetition when defining mode and
    ## state access methods with and without side-effects. The micromotion operator
    ## caching is determined through the parameter function pfunc
    #U0 = modes(fb, Val(true))
    #if t == zero(t)
        #return U0
    #else
        #Ut = pfunc(fb, t; kwargs...).data
        #return Ut * U0
    #end
#end

######## Get States and Modes at t>0
## No side-effects

function states(
    fb::FloquetBasis,
    tlist::AbstractVector,
    ::Val{true};
    kwargs...
    )
    return _states(fb, tlist, propagator; kwargs...)
end

function states(
    fb::FloquetBasis,
    tlist::AbstractVector,
    ::Val{false} = Val(false);
    kwargs...
    )
    return [_data_to_ketlist(ψt, fb.dims)
            for ψt in _states(fb, tlist, propagator; kwargs...)]
end


function states(fb::FloquetBasis, t::Real, ::Val{true}; kwargs...)
    return _states(fb, t, propagator; kwargs...)
end

function states(fb::FloquetBasis, t::Real, ::Val{false}=Val(false); kwargs...)
    return _states(fb, t, propagator; kwargs...) |>
        M -> _data_to_ketlist(M, fb.dims)
end

function modes(
    fb::FloquetBasis,
    tlist::AbstractVector,
    ::Val{true};
    kwargs...)
    state_mtrxs = states(fb, tlist, Val(true); kwargs...)
    return [_state_mtrx_to_mode(ψt, fb.equasi, t)
            for (t, ψt) in zip(tlist, state_mtrxs)]
end

function modes(
    fb::FloquetBasis,
    tlist::AbstractVector,
    ::Val{false}=Val(false);
    kwargs...
    )
    return [_data_to_ketlist(ut, fb.dims)
            for ut in modes(fb, tlist, Val(true); kwargs...)]
end

function modes(fb::FloquetBasis, t::Real, ::Val{true}; kwargs...)
    return states(fb, t, Val(true); kwargs...) |>
        M -> _state_mtrx_to_mode(M, fb.equasi, t)
end

function modes(fb::FloquetBasis, t::Real, ::Val{false}=Val(false); kwargs...)
    return modes(fb, t, Val(true); kwargs...) |>
        M -> _data_to_ketlist(M, fb.dims)
end

## Side effect: Cache previously uncalculated micromotion operators to FloquetBasis
##
##
function states!(
    fb::FloquetBasis,
    tlist::AbstractVector,
    ::Val{true};
    kwargs...
    )
    return _states(fb, tlist, propagator!; kwargs...)
end

function states!(
    fb::FloquetBasis,
    tlist::AbstractVector,
    ::Val{false} = Val(false);
    kwargs...
    )
    return [_data_to_ketlist(ψt, fb.dims)
            for ψt in _states(fb, tlist, propagator!; kwargs...)]
end

function states!(fb::FloquetBasis, t::Real, ::Val{true}; kwargs...)
    return _states(fb, t, propagator!; kwargs...)
end

function states!(fb::FloquetBasis, t::Real, ::Val{false}=Val(false); kwargs...)
    return _states(fb, t, propagator!; kwargs...) |>
        M -> _data_to_ketlist(M, fb.dims)
end

function modes!(fb::FloquetBasis, t::Real, ::Val{true}; kwargs...)
    return states!(fb, t, Val(true); kwargs...) |>
        M -> _state_mtrx_to_mode(M, fb.equasi, t)
end

function modes!(fb::FloquetBasis, t::Real, ::Val{false}=Val(false); kwargs...)
    return modes!(fb, t, Val(true); kwargs...) |>
        M -> _data_to_ketlist(M, fb.dims)
end

############## From and to Floquet Basis funcs
### For consitency with qutip, define behavior for both Matrix and Qobj data types,

# First, define functions to return transformation matrices to and from the Floquet
# basis. These will just be wrappers around mode and state, since calling
# these functions with the ::Val{true} set will return the transformation matrix
# in .data format

# no caching new micromotion operators
function from_floquet_basis(
    fb::FloquetBasis,
    t::Real=0.0,
    ::Val{false}=Val(false);
    mode_basis::Bool=false,
    kwargs...
    )
    bfunc = mode_basis ? modes : states
    return bfunc(fb, t, Val(true); kwargs...) |>
        U -> Qobj(U, dims=fb.dims)
end

function from_floquet_basis(
    fb::FloquetBasis,
    t::Real,
    ::Val{true};
    mode_basis::Bool=false,
    kwargs...)
    bfunc= mode_basis ? modes : states
    return bfunc(fb, t, Val(true); kwargs...)
end


function to_floquet_basis(
    fb::FloquetBasis,
    t::Real=0.0,
    ::Val{false}=Val(false);
    mode_basis::Bool=false,
    kwargs...
    )
    # to is just adjoint of from
    return from_floquet_basis(fb,
                              t,
                              Val(false);
                              mode_basis=mode_basis,
                              kwargs...)'
end

function to_floquet_basis(
    fb::FloquetBasis,
    t::Real,
    ::Val{true};
    mode_basis::Bool=false,
    kwargs...)
    # to is just adjoint of from
    return from_floquet_basis(fb,
                              t,
                              Val(true);
                              mode_basis=mode_basis,
                              kwargs...)'
end

# with micromotion operator caching

function from_floquet_basis!(
    fb::FloquetBasis,
    t::Real=0.0,
    ::Val{false}=Val(false);
    mode_basis::Bool=false,
    kwargs...
    )
    bfunc = mode_basis ? modes! : states!
    return bfunc(fb, t, Val(true); kwargs...) |>
        U -> Qobj(U, dims=fb.dims)
end

function from_floquet_basis!(
    fb::FloquetBasis,
    t::Real,
    ::Val{true};
    mode_basis::Bool=false,
    kwargs...)
    bfunc= mode_basis ? modes! : states!
    return bfunc(fb, t, Val(true); kwargs...)
end


function to_floquet_basis!(
    fb::FloquetBasis,
    t::Real=0.0,
    ::Val{false}=Val(false);
    mode_basis::Bool=false,
    kwargs...
    )
    # to is just adjoint of from
    return from_floquet_basis!(fb,
                              t,
                              Val(false);
                              mode_basis=mode_basis,
                              kwargs...)'
end

function to_floquet_basis!(
    fb::FloquetBasis,
    t::Real,
    ::Val{true};
    mode_basis::Bool=false,
    kwargs...)
    # to is just adjoint of from
    return from_floquet_basis!(fb,
                              t,
                              Val(true);
                              mode_basis=mode_basis,
                              kwargs...)'
end


# now define functions that apply the transformation to a given
# operator or vector

##### for Qobj return type
## no micromotion caching
function from_floquet_basis(
    fb::FloquetBasis,
    floquet_qobj::AbstractQuantumObject,
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = from_floquet_basis(fb,
                                 t,
                                 Val(false);
                                 mode_basis=mode_basis,
                                 kwargs...)
    if isket(floquet_qobj)
        lab_qobj = U_trans * floquet_qobj
    elseif isbra(floquet_qobj)
        lab_qobj = floquet_qobj * U_trans'
    elseif floquet_qobj.type==Operator() # isop returns false for QobjEvo
        lab_qobj = U_trans * floquet_qobj * U_trans'
    else
        throw(
            ErrorException(
                "FloquetBasis transformations not supported for operators of type $floquet_qobj.type"
            )
        )
    end
    return lab_qobj
end


function to_floquet_basis(
    fb::FloquetBasis,
    lab_qobj::AbstractQuantumObject,
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = to_floquet_basis(fb,
                               t,
                               Val(false);
                               mode_basis=mode_basis,
                               kwargs...)
    if isket(lab_qobj)
        floquet_qobj = U_trans * lab_qobj
    elseif isbra(lab_qobj)
        floquet_qobj = lab_qobj * U_trans'
    elseif lab_qobj.type==Operator() # isop returns false for QobjEvo
        floquet_qobj = U_trans * lab_qobj * U_trans'
    else
        throw(
            ErrorException(
                "FloquetBasis transformations not supported for operators of type $lab_qobj.type"
            )
        )
    end
    return floquet_qobj
end

### with micromotion caching

function from_floquet_basis!(
    fb::FloquetBasis,
    floquet_qobj::AbstractQuantumObject,
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = from_floquet_basis!(fb,
                                 t,
                                 Val(false);
                                 mode_basis=mode_basis,
                                 kwargs...)
    if isket(floquet_qobj)
        lab_qobj = U_trans * floquet_qobj
    elseif isbra(floquet_qobj)
        lab_qobj = floquet_qobj * U_trans'
    elseif floquet_qobj.type==Operator() # isop returns false for QobjEvo
        lab_qobj = U_trans * floquet_qobj * U_trans'
    else
        throw(
            ErrorException(
                "FloquetBasis transformations not supported for operators of type $floquet_qobj.type"
            )
        )
    end
    return lab_qobj
end


function to_floquet_basis!(
    fb::FloquetBasis,
    lab_qobj::AbstractQuantumObject,
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = to_floquet_basis!(fb,
                               t,
                               Val(false);
                               mode_basis=mode_basis,
                               kwargs...)
    if isket(lab_qobj)
        floquet_qobj = U_trans * lab_qobj
    elseif isbra(lab_qobj)
        floquet_qobj = lab_qobj * U_trans'
    elseif lab_qobj.type==Operator() # isop returns false for QobjEvo
        floquet_qobj = U_trans * lab_qobj * U_trans'
    else
        throw(
            ErrorException(
                "FloquetBasis transformations not supported for operators of type $lab_qobj.type"
            )
        )
    end
    return floquet_qobj
end


################# for .data return type
### without micromotion caching

function from_floquet_basis(
    fb::FloquetBasis,
    floquet_array::Union{AbstractMatrix, AbstractVector},
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = from_floquet_basis(fb,
                                 t,
                                 Val(true);
                                 mode_basis=mode_basis,
                                 kwargs...)
    if floquet_array isa AbstractVector
        lab_array = U_trans * floquet_array
    else
        lab_array = U_trans * floquet_array * U_trans'
    end
    return lab_array
end


function to_floquet_basis(
    fb::FloquetBasis,
    lab_array::Union{AbstractMatrix, AbstractVector},
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = to_floquet_basis(fb,
                                 t,
                                 Val(true);
                                 mode_basis=mode_basis,
                                 kwargs...)
    if lab_array isa AbstractVector
        floquet_array = U_trans * lab_array
    else
        floquet_array = U_trans * lab_array * U_trans'
    end
    return floquet_array
end

# with micromotion caching

function from_floquet_basis!(
    fb::FloquetBasis,
    floquet_array::Union{AbstractMatrix, AbstractVector},
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = from_floquet_basis!(fb,
                                 t,
                                 Val(true);
                                 mode_basis=mode_basis,
                                 kwargs...)
    if floquet_array isa AbstractVector
        lab_array = U_trans * floquet_array
    else
        lab_array = U_trans * floquet_array * U_trans'
    end
    return lab_array
end


function to_floquet_basis!(
    fb::FloquetBasis,
    lab_array::Union{AbstractMatrix, AbstractVector},
    t::Real=0.0;
    mode_basis::Bool=false,
    kwargs...
    )
    U_trans = to_floquet_basis!(fb,
                                 t,
                                 Val(true);
                                 mode_basis=mode_basis,
                                 kwargs...)
    if lab_array isa AbstractVector
        floquet_array = U_trans * lab_array
    else
        floquet_array = U_trans * lab_array * U_trans'
    end
    return floquet_array
end
