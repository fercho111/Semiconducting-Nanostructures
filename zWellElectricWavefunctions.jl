include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using HDF5
using QuadGK
using Unitful
using LinearAlgebra
using Printf

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60
const eF = 0.4   # meV/nm

V(z) = PotentialWell.potential_well(z) + eF * z

N = 50
H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V, L)

eigvals, eigvecs = eigen(H)

println("First 5 energies (meV):")
println(eigvals[1:5])

z_grid = -L/2:0.01:L/2

# Prepare output file
filename = @sprintf("QW_asym_L%.1f_N%d_states%d.dat", L, N, 4)
open(filename, "w") do io
    # Write header with metadata
    println(io, "# File: $filename")
    println(io, "# System: Asymmetric quantum well potential with E-field")
    println(io, "# Params: L = $(L) nm, N = $N basis states")
    @printf(io, "# Energies (meV): E1=%.6f, E2=%.6f, E3=%.6f, E4=%.6f\n",
            eigvals[1], eigvals[2], eigvals[3], eigvals[4])
    println(io, "# Columns: z_nm   V_meV   psi1_sq   psi2_sq   psi3_sq   psi4_sq")

    # Write data rows
    for (i, z) in enumerate(z_grid)
        Vval = V(z)
        ψsq = [abs2(QuantumWell.wf_realspace(eigvecs[:,k], QuantumWell.psi_inf, L)(z))
               for k in 1:4]
        @printf(io, "%10.4f\t%10.6f\t%12.6e\t%12.6e\t%12.6e\t%12.6e\n",
                z, Vval, ψsq[1], ψsq[2], ψsq[3], ψsq[4])
    end
end

println("Data written to $filename")
