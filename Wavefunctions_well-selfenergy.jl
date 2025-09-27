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

const eF = 4 # meV/nm

# Coulomb prefactor in meV·nm
# Start with J·m, convert to eV·nm, then to meV·nm
const e2_4pie0_Jm = Constants.electron^2 / (4 * π * Constants.epsilon_0)  # J·m
const e2_4pie0_meVnm = (e2_4pie0_Jm / Constants.electron) * 1e3 * 1e9     # meV·nm

# Self-energy potential due to dielectric mismatch (output in meV)
function selfenergy(z::Real; ε_in=12.9, ε_out=1.0, R=L/2, kmax=20)
    r = abs(z)                 # in 1D we take |z|
    if r > R
        return 0.0             # outside dot, no self-energy correction
    end
    s = 0.0
    for k in 0:kmax
        num = (k+1) * (ε_in - ε_out)
        den = ε_out*(k+1) + ε_in*k
        s += (num/den) * (r/R)^(2k)
    end
    return (e2_4pie0_meVnm / (2 * ε_in * R)) * s  # meV
end


V(z) = PotentialWell.potential_well(z) + selfenergy(z, ε_in=12.35, ε_out=3.2, R=L/2, kmax=50)

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

savefig("QW4.png")