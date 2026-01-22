using CairoMakie
using LinearAlgebra

N = 10
a = destroy(N)
a_d = a'
H = num(N) + (a + a_d)
Ht = QobjEvo(H, (p, t) -> cos(t)) # for test throw
T = 2π
psi0 = fock(N, 3)
t_l = LinRange(0, 200, 1000)
fb_test1 = FloquetBasis(Ht, T)