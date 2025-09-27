include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using QuadGK
using Unitful
using LinearAlgebra
using DelimitedFiles

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60
const N = 200

# --- Coulomb prefactor in meV·nm ---
const e2_4pie0_Jm    = Constants.electron^2 / (4 * π * Constants.epsilon_0)  # J·m
const e2_4pie0_meVnm = (e2_4pie0_Jm / Constants.electron) * 1e3 * 1e9        # meV·nm

# --- Geometry of the inner region (asymmetric well) ---
const zmin = -7.0
const zmax =  3.0
const c    = (zmin + zmax) / 2   # center = -2 nm
const Rin  = (zmax - zmin) / 2   # effective radius = 5 nm
const EPS  = 1e-12               # nm, to avoid r=0 divisions

# --- Self-energy (meV) ---
function selfenergy(z::Real; ε_in=12.35, ε_out=3.2, kmax=50)
    r = abs(z - c) + EPS
    if r <= Rin + 1e-15
        s = 0.0
        for k in 0:kmax
            num = (k + 1) * (ε_in - ε_out)
            den = ε_out * (k + 1) + ε_in * k
            s += (num/den) * (r/Rin)^(2k)
        end
        return (e2_4pie0_meVnm / (2 * ε_in * Rin)) * s
    else
        s = 0.0
        for k in 1:kmax
            num = k * (ε_out - ε_in)
            den = ε_out * (k + 1) + ε_in * k
            s += (num/den) * (Rin/r)^(2(k+1))
        end
        return (e2_4pie0_meVnm / (2 * ε_out * Rin)) * s
    end
end

# --- Potential with self-energy + electric field ---
const eF = -4  # meV/nm = -40 kV/cm
V_selfE(z) = PotentialWell.potential_well(z) + selfenergy(z) + eF * z

# --- Solve Hamiltonian ---
H_selfE = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V_selfE, L)
eigvals_selfE, eigvecs_selfE = eigen(H_selfE)

println("With self-energy + E-field energies (meV): ", eigvals_selfE[1:2])

# --- Grid ---
z_grid = collect(-L/2:0.06:L/2)

# --- Evaluate potential ---
V_vals = V_selfE.(z_grid)

# --- First 2 states (offset at energies, scaled) ---
scale_factor = 500
ψ_data = Array{Float64}(undef, length(z_grid), 2)

for k in 1:2
    ψk = QuantumWell.wf_realspace(eigvecs_selfE[:,k], QuantumWell.psi_inf, L)
    ψ_vals = abs2.(ψk.(z_grid))
    ψ_data[:,k] = scale_factor * ψ_vals .+ eigvals_selfE[k]
end

# --- Save to .dat file ---
header = "# z (nm)   Potential (meV)   ψ1^2 offset (meV)   ψ2^2 offset (meV)"
data = [z_grid V_vals ψ_data]
open("QW_selfE_Efield.dat", "w") do io
    println(io, header)
    writedlm(io, data)
end
