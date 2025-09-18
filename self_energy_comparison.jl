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

const L = 60
const N = 50

# --- Coulomb prefactor in meV·nm ---
const e2_4pie0_Jm = Constants.electron^2 / (4 * π * Constants.epsilon_0)  # J·m
const e2_4pie0_meVnm = (e2_4pie0_Jm / Constants.electron) * 1e3 * 1e9     # meV·nm

# --- Self-energy potential due to dielectric mismatch (output in meV) ---
function selfenergy(z::Real; ε_in=12.35, ε_out=3.2, R=L/2, kmax=100)
    r = abs(z)                 
    if r > R
        return 0.0 # outside dot, no self-energy correction
    end
    s = 0.0
    for k in 0:kmax
        num = (k+1) * (ε_in - ε_out)
        den = ε_out*(k+1) + ε_in*k
        s += (num/den) * (r/R)^(2k)
    end
    return (e2_4pie0_meVnm / (2 * ε_in * R)) * s  # meV
end

# --- Two versions of the potential ---
V_bare(z) = PotentialWell.potential_well(z)
V_self(z) = PotentialWell.potential_well(z) + selfenergy(z)

# --- Solve Hamiltonians ---
H_bare = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V_bare, L)
eigvals_bare, eigvecs_bare = eigen(H_bare)

H_self = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V_self, L)
eigvals_self, eigvecs_self = eigen(H_self)

println("Bare well energies (meV): ", eigvals_bare[1:2])
println("With self-energy energies (meV): ", eigvals_self[1:2])

# --- Grid for plotting ---
z_grid = -L/2:0.05:L/2

# --- Plot the potentials ---
plot(z_grid, V_bare.(z_grid), label="Bare well", lw=2, color=:black)
plot!(z_grid, V_self.(z_grid), label="Well + self-energy", lw=2, color=:red, ls=:dash)

# --- Overlay first 2 states for both cases ---
scale_factor = 500  # arbitrary scaling for visibility

for k in 1:2
    ψ_bare = QuantumWell.wf_realspace(eigvecs_bare[:,k], QuantumWell.psi_inf, L)
    ψ_self = QuantumWell.wf_realspace(eigvecs_self[:,k], QuantumWell.psi_inf, L)

    ψ_bare_vals = abs2.(ψ_bare.(z_grid))
    ψ_self_vals = abs2.(ψ_self.(z_grid))

    # Offset at the eigenenergy
    plot!(z_grid, scale_factor*ψ_bare_vals .+ eigvals_bare[k], 
          label="Bare n=$k", color=:blue, lw=2)

    plot!(z_grid, scale_factor*ψ_self_vals .+ eigvals_self[k], 
          label="Self-energy n=$k", color=:orange, lw=2, ls=:dash)
end

xlabel!("z (nm)")
ylabel!("Energy (meV)")
title!("Comparison of bare vs. self-energy corrected states")

savefig("QW_compare.png")
