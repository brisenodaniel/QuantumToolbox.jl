using CairoMakie
using LinearAlgebra
using Test

N = 10
a = destroy(N)
a_d = a'
H = num(N) + (a + a_d)
Ht = QobjEvo(H, (p, t) -> cos(t)) # for test throw
T = 2π
psi0 = fock(N, 3)
t_l = LinRange(0, 200, 1000)
fb_test1 = FloquetBasis(Ht, T)
floquet_psi0 = to_floquet_basis(fb_test1, psi0, 0.0)

states = sesolve(Ht, psi0, t_l).states

states_fse = fsesolve(fb_test1, psi0, t_l)


for (t, state) in zip(tlist, states)
    from_floquet = from_floquet_basis(fb_test1, floquet_psi0, t)
    ov = abs(state' * from_floquet)
#    @test isapprox(ov, 1.0; atol=8e-5)
end