include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using HDF5
using Plots
using QuadGK
using Unitful
using LinearAlgebra

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60

# to go from V/m to meV/nm, we multiply the whole -eF term together, to get eV / m
# eV / m = 1000 meV / 1 eV * 1 m / 1e9 nm = 1e6 meV / nm
# then 4e6 V/m corresponds to 4 meV / nm

const eF = 0.4 # meV/nm

V(z) = PotentialWell.potential_well(z) + eF * z


N = 50
H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V, L)

eigvals, eigvecs = eigen(H)

# Print first 5 energies
println("First 5 energies (meV):")
println(eigvals[1:5])

z_grid = -L/2:0.01:L/2

# plot the well
plot(z_grid, V.(z_grid), label="Potential+Electric Field", lw=2, color=:black)

# Overlay first 4 states
scale_factor = 500  # arbitrary vertical scaling for visibility
for k in 1:4
    ψk = QuantumWell.wf_realspace(eigvecs[:,k], QuantumWell.psi_inf, L)  # callable ψ(z)
    ψ_vals = abs2.(ψk.(z_grid))  # evaluate on grid
    plot!(z_grid, scale_factor * ψ_vals .+ eigvals[k], label="n=$k")
end

xlabel!("z (nm)")
ylabel!("Energy (meV)")

savefig("QW2.png")