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

# Sweep E_field from 0 to 4 in 0.1 increments
eF_sweep = 0:0.1:4

# Arrays for scaled energy differences
ΔE_21 = Float64[]
ΔE_31 = Float64[]
ΔE_41 = Float64[]

for eF in eF_sweep
    V(z) = PotentialWell.potential_well(z) + eF * z

    H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V, L)
    eigvals, eigvecs = eigen(H)

    # Reference energy (ground state)
    E1 = eigvals[1]

    # Scaled differences
    push!(ΔE_21, (eigvals[2] - E1) / 1)
    push!(ΔE_31, (eigvals[3] - E1) / 2)
    push!(ΔE_41, (eigvals[4] - E1) / 3)
end

# Plot scaled differences
plot(eF_sweep, ΔE_21, label="(E₂ - E₁)/1", xlabel="E_field (meV/nm)", ylabel="Scaled ΔE (meV)", legend=:topright)
plot!(eF_sweep, ΔE_31, label="(E₃ - E₁)/2")
plot!(eF_sweep, ΔE_41, label="(E₄ - E₁)/3")

savefig("quantum_well_scaled_differences.png")
