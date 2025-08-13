using Unitful
import PhysicalConstants.CODATA2022 as C
using QuadGK
using LinearAlgebra
using Plots
using HDF5

eigvals = h5read("wf_data.h5", "eigvals")
eigvecs = h5read("wf_data.h5", "eigvecs")

# en nanometros
const L = 60.0
const Z_1 = -11.0
const Z_2 = -7.0
const Z_3 = 3.0
const Z_4 = 7.0
const Z_5 = 15.0

# datos del problema, todos en SI
const electron = 1.60217663e-19
const hbar = 1.05457182e-34
const m_e = 9.1093837e-31
const m_eff = 0.067 * m_e
const nm = 1e-9

# definimos el potencial multipaso
function V(z::Real)
    if -L/2 <= z <= Z_1
        return 282.8
    elseif Z_1 < z <= Z_2
        return 101.1
    elseif Z_2 < z <= Z_3
        return 0.0
    elseif Z_3 < z <= Z_4
        return 41.1
    elseif Z_4 < z <= Z_5
        return 151.2
    elseif Z_5 < z <= L/2
        return 212.3
    end
end

# definimos la base del pozo infinito
psi_inf(n::Int, z::Real) = sqrt(2/L) * sin(n * π * (z + L/2) / L)
E_inf(n::Int) = (n * π / (L*nm))^2 * hbar^2 / (2 * m_eff) / electron * 1000 # meV


# z-grid for plotting
z_grid = range(-L/2, L/2, length=1000)

function wf_realspace(coeffs, basis_func, zgrid)
    # Sum over basis functions with the given coefficients
    [sum(coeffs[n] * basis_func(n, z) for n in 1:length(coeffs)) for z in zgrid]
end

# Convert energies to meV
energies = eigvals

# Potential over the same grid
V_grid = [V(z) for z in z_grid]

# Plot potential first
plot(z_grid, V_grid, label="Potential", lw=2, color=:black)

# Overlay first 4 states
scale_factor = 500  # arbitrary vertical scaling for visibility
for k in 1:4
    ψ_k = abs2.(wf_realspace(eigvecs[:,k], psi_inf, z_grid))
    plot!(z_grid, scale_factor * ψ_k .+ energies[k], label="n=$k")
end

xlabel!("z (nm)")
ylabel!("Energy (meV)")

savefig("quantum_well_states.png")
