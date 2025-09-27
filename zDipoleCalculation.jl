# ===== Export dipoles & energies vs field to a .dat file (no plots) =====
include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using QuadGK
using LinearAlgebra
using Printf

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60
const N = 50

# Real-space dipole matrix element <m|z|n>
function dipole_element(coeffs_m::AbstractVector, coeffs_n::AbstractVector, basis_func, L::Real)
    ψm = QuantumWell.wf_realspace(coeffs_m, basis_func, L)
    ψn = QuantumWell.wf_realspace(coeffs_n, basis_func, L)
    val, _ = quadgk(z -> ψm(z) * z * ψn(z), -L/2, L/2)
    return val
end

# Sweep of the linear field term eF (units: meV/nm) used in V(z) = V0(z) + eF*z
eF_sweep = 0:0.1:4

# ----- Output file (explicit name) -----
filename = "QW_dipoles_vs_field.dat"
open(filename, "w") do io
    # ---- Header (commented; safe for Origin) ----
    println(io, "# File: QW_dipoles_vs_field.dat")
    println(io, "# System: Asymmetric quantum well with linear term eF·z in the potential")
    println(io, "# Note: eF is in meV/nm (so V(z) is in meV).")
    @printf(io, "# Sweep: E_field from %.1f to %.1f meV/nm in steps of %.1f meV/nm\n",
            first(eF_sweep), last(eF_sweep), step(eF_sweep))
    println(io, "# Columns:")
    println(io, "#  1: E_field_meV_per_nm")
    println(io, "#  2: E0_meV   3: E1_meV   4: E2_meV   5: E3_meV")
    println(io, "#  6: z10_nm   7: z20_nm   8: z30_nm   (dipole elements ⟨1|z|0⟩, ⟨2|z|0⟩, ⟨3|z|0⟩)")
    println(io, "#  9: abs_z10_nm  10: abs_z20_nm  11: abs_z30_nm")
    println(io, "# ---------------------------------------------------------------")

    # ---- Sweep and write rows ----
    for eF in eF_sweep
        V(z) = PotentialWell.potential_well(z) + eF * z

        H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V, L)
        eigvals, eigvecs = eigen(H)

        # Make eigenvector signs consistent (optional, for smooth dipoles)
        for j in 1:size(eigvecs, 2)
            if eigvecs[1, j] < 0
                eigvecs[:, j] .= -eigvecs[:, j]
            end
        end

        # Dipole transitions: ⟨1|z|0⟩, ⟨2|z|0⟩, ⟨3|z|0⟩
        z10 = dipole_element(eigvecs[:, 2], eigvecs[:, 1], QuantumWell.psi_inf, L)
        z20 = dipole_element(eigvecs[:, 3], eigvecs[:, 1], QuantumWell.psi_inf, L)
        z30 = dipole_element(eigvecs[:, 4], eigvecs[:, 1], QuantumWell.psi_inf, L)

        @printf(io, "%8.3f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\n",
                eF,              # 1: E_field_meV_per_nm
                eigvals[1],      # 2: E0_meV
                eigvals[2],      # 3: E1_meV
                eigvals[3],      # 4: E2_meV
                eigvals[4],      # 5: E3_meV
                z10, z20, z30,   # 6–8: signed dipoles (nm)
                abs(z10), abs(z20), abs(z30))  # 9–11: magnitudes (nm)
    end
end

println("Data written to $(filename)")
