using Unitful
import PhysicalConstants.CODATA2022 as C
using QuadGK
using LinearAlgebra
using Plots
using HDF5
include("multistep.jl")

# datos del problema, todos en SI
const electron = 1.60217663e-19
const hbar = 1.05457182e-34
const m_e = 9.1093837e-31
const m_eff = 0.067 * m_e
const nm = 1e-9


# definimos la base del pozo infinito
psi_inf(n::Int, z::Real) = sqrt(2/L) * sin(n * π * (z + L/2) / L)
E_inf(n::Int) = (n * π / (L*nm))^2 * hbar^2 / (2 * m_eff) / electron * 1000 # meV

function build_hamiltonian(N, ψ, E_kin, V)
    H = zeros(N, N)
    for m in 1:N
        for n in 1:N
            if m == n
                # Kinetic part is diagonal for infinite well basis
                H[m,n] += E_kin(n)
            end
            # Potential matrix element: ⟨m|V|n⟩
            val, _ = quadgk(z -> ψ(m, z)*V(z)*ψ(n, z), -L/2, L/2)
            H[m,n] += val
        end
    end
    return H
end

# test
N = 50
H = build_hamiltonian(N, psi_inf, E_inf, potential_well)

eigvals, eigvecs = eigen(H)

# Print first 5 energies
#println("First 5 energies (meV):")
#println(eigvals[1:5])
#
## Save
#h5write("wf_data.h5", "eigvals", eigvals)
#h5write("wf_data.h5", "eigvecs", eigvecs)
