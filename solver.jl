include("Constants.jl")
include("QuantumWell.jl")
include("PotentialWell.jl")

using HDF5
using Plots
using QuadGK
using Unitful
using LinearAlgebra

using .QuantumWell
using .PotentialWell
using .Constants

# datos del problema, todos en SI

const L = 60.0 

N = 50
H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, PotentialWell.potential_well, L)

eigvals, eigvecs = eigen(H)

# Print first 5 energies
println("First 5 energies (meV):")
println(eigvals[1:5])

z_grid = -L/2:0.01:L/2

# plot the well
plot(z_grid, PotentialWell.potential_well.(z_grid), label="Potential", lw=2, color=:black)

# Overlay first 4 states
scale_factor = 500  # arbitrary vertical scaling for visibility
for k in 1:4
    ψk = QuantumWell.wf_realspace(eigvecs[:,k], QuantumWell.psi_inf, L)  # callable ψ(z)
    ψ_vals = abs2.(ψk.(z_grid))  # evaluate on grid
    plot!(z_grid, scale_factor * ψ_vals .+ eigvals[k], label="n=$k")
end


xlabel!("z (nm)")
ylabel!("Energy (meV)")

savefig("QW1.png")