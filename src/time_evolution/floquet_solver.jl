export FloquetBasis, FloquetEvolutionSol, propagator, propagator!, fsesolve, fsesolve!, to_floquet_basis, to_floquet_basis!, from_floquet_basis, from_floquet_basis!, modes, modes!, states, states!
# script helper functions
function _to_period_interval(tlist::AbstractVector, T::Real)
    # function maps all elements ``t`` in `tlist` outside the interval ``[0, T)`` to an equivalent
    # time ``\tau`` such that ``mod(t, T) = \tau``
    if !isempty(tlist)
        tlist = mod.(tlist, T) |> unique |> sort
    end
    return tlist
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


struct FloquetEvolutionSol{
    TT1<:AbstractVector{<:Real},
    TT2<:AbstractVector{<:Real},
    TS<:AbstractVector{<:AbstractQuantumObject},
    AlgT<:AbstractODEAlgorithm,
    TolT<:Real
}
    times::TT1
    times_states::TT2
    states::TS
    expect::AbstractVector{ComplexF64}
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
    TQ<:AbstractVector{<:AbstractQuantumObject},
    PP,
}
    H::AbstractQuantumObject
    T::Float64
    precompute::AbstractVector{Float64}
    U_T::Qobj
    Ulist::TQ
    equasi::AbstractVector{Float64}
    alg::AbstractODEAlgorithm
    abstol_UT::Float64
    reltol_UT::Float64
    params::PP
    kwargs::Union{Dict, NamedTuple}

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
        precompute::Union{AbstractVector{<:Real}, Nothing}=nothing;
        params::PP=nothing,
        alg::AbstractODEAlgorithm = Vern7(lazy = false),
        reltol_UT::Real=1e-12, # smaller tolerance for error when computing period-prop
        abstol_UT::Union{Real, Nothing}=nothing, # keep error floor at default unless user specifies otherwise
        tidyup_UT::Bool=true, # set noise floor of period-propagator at abstol of solver
        tidyup_micromotion::Bool=false, # set noise floor of micromotion operators at abstol of solver
        kwargs...
        ) where {PP}
        # create editable version of kwargs
        kwargs = Dict{Symbol, Any}(kwargs)
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
            saveat=tlist,
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
        period_phases = eigenenergies(U_T)
        equasi = angle.(period_phases) ./ T |> sort

        new{typeof(Ulist), PP}(
            H,
            Float64(T),
            precompute,
            U_T,
            Ulist,
            equasi,
            sol.alg,
            sol.abstol,
            sol.reltol,
            params,
            kwargs,
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
    tlist::AbstractVector{<:Real},
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


function is_memoized(fb::FloquetBasis, t::Real)
    return t==zero(t) || Float64(t)==fb.T || mod(t, fb.T) ∈ fb.precompute
end

function _from_micromotion_cache(fb::FloquetBasis, t::Real)
    if is_memoized(fb, t)
        t = mod(t, fb.T)
        return t==zero(t) ?
            qeye_like(fb) : fb.Ulist[findfirst(==(t), fb.precompute)]
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
    return ϕ_mat * M
end

######## Get  modes at time 0

function modes(fb::FloquetBasis, ::Val{true}; kwargs...)
    _, _, U0 = eigenstates(fb.U_T)
    return _state_mtrx_to_mode(U0, fb.equasi, fb.T)
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
    tlist::AbstractVector{<:Real},
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
    tlist::AbstractVector{<:Real},
    ::Val{true};
    kwargs...
    )
    return _states(fb, tlist, propagator; kwargs...)
end

function states(
    fb::FloquetBasis,
    tlist::AbstractVector{<:Real},
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
    tlist::AbstractVector{<:Real},
    ::Val{true};
    kwargs...)
    state_mtrxs = states(fb, tlist, Val(true); kwargs...)
    return [_state_mtrx_to_mode(ψt, fb.equasi, t)
            for (t, ψt) in zip(tlist, state_mtrxs)]
end

function modes(
    fb::FloquetBasis,
    tlist::AbstractVector{<:Real},
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
    tlist::AbstractVector{<:Real},
    ::Val{true};
    kwargs...
    )
    return _states(fb, tlist, propagator!; kwargs...)
end

function states!(
    fb::FloquetBasis,
    tlist::AbstractVector{<:Real},
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
