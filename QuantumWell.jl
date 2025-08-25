module QuantumWell

using Unitful
import PhysicalConstants.CODATA2022 as C
using QuadGK
using LinearAlgebra

# ----------------------------
# Physical constants (SI units)
# ----------------------------
const electron = 1.60217663e-19
const hbar     = 1.05457182e-34
const m_e      = 9.1093837e-31
const nm       = 1e-9

# Effective mass (e.g. GaAs)
const m_eff    = 0.067 * m_e

# ----------------------------
# Infinite square well basis
# ----------------------------
# NOTE: length L should be provided in nm
# We pass it explicitly to functions for modularity
function psi_inf(n::Int, z::Real, L::Real)
    return sqrt(2/L) * sin(n * π * (z + L/2) / L)
end

function E_inf(n::Int, L::Real)
    return (n * π / (L*nm))^2 * hbar^2 / (2 * m_eff) / electron * 1000  # in meV
end

# ----------------------------
# Hamiltonian builder
# ----------------------------
function build_hamiltonian(N::Int, ψ, E_kin, V::Function, L::Real)
    H = zeros(Float64, N, N)
    for m in 1:N
        for n in 1:N
            if m == n
                H[m,n] += E_kin(n, L)
            end
            val, _ = quadgk(z -> ψ(m, z, L) * V(z) * ψ(n, z, L), -L/2, L/2)
            H[m,n] += val
        end
    end
    return Symmetric(H)
end

function wf_realspace(coeffs::AbstractVector, basis_func, L::Real)
    return z -> sum(coeffs[n] * basis_func(n, z, L) for n in 1:length(coeffs))
end

# ----------------------------
# Exports
# ----------------------------
export psi_inf, E_inf, build_hamiltonian

end # module
