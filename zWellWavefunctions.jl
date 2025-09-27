include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using DelimitedFiles
using LinearAlgebra
using Printf

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60.0
const N = 50

# Build Hamiltonian and solve
H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, PotentialWell.potential_well, L)
eigvals, eigvecs = eigen(H)

# Spatial grid
z_grid = collect(-L/2:0.01:L/2)

# Potential on grid
V_vals = PotentialWell.potential_well.(z_grid)

# First 4 states: |ψ_k(z)|^2 (no energy offset, as requested)
num_states = 4
psi_mat = Array{Float64}(undef, length(z_grid), num_states)
for k in 1:num_states
    ψk = QuantumWell.wf_realspace(eigvecs[:, k], QuantumWell.psi_inf, L)  # callable ψ(z)
    psi_mat[:, k] = abs2.(ψk.(z_grid))
end

# Assemble table: z | V(z) | ψ1^2 | ψ2^2 | ψ3^2 | ψ4^2
data = [z_grid V_vals psi_mat]

# Descriptive filename
filename = @sprintf "QW_asym_L%.1f_N%d_states%d.dat" L N num_states

# Write with a clear header (Origin will ignore lines starting with '#')
open(filename, "w") do io
    @printf(io, "# File: %s\n", filename)
    @printf(io, "# System: Asymmetric quantum well potential only (no self-energy, no E-field)\n")
    @printf(io, "# Params: L = %.3f nm, N = %d basis states\n", L, N)
    @printf(io, "# Energies (meV): E1=%.6f, E2=%.6f, E3=%.6f, E4=%.6f\n",
            eigvals[1], eigvals[2], eigvals[3], eigvals[4])
    println(io, "# Columns: z_nm   V_meV   psi1_sq   psi2_sq   psi3_sq   psi4_sq")
    writedlm(io, data)
end

println("Saved: ", filename)
