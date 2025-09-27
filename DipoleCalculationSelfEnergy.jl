include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using HDF5
using Plots
using QuadGK
using Unitful
using LinearAlgebra
using LaTeXStrings

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60

# self energy potential

# --- Coulomb prefactor in meV·nm ---
const e2_4pie0_Jm = Constants.electron^2 / (4 * π * Constants.epsilon_0)  # J·m
const e2_4pie0_meVnm = (e2_4pie0_Jm / Constants.electron) * 1e3 * 1e9     # meV·nm

# --- Geometry of the inner region (asymmetric well) ---
const zmin = -7.0
const zmax =  3.0
const c    = (zmin + zmax) / 2   # center = -2 nm
const Rin  = (zmax - zmin) / 2   # effective radius = 5 nm

const EPS = 1e-12  # nm, to avoid r=0 divisions

# Self-energy (meV) using radius from region center; well-behaved at boundaries
function selfenergy(z::Real; ε_in=12.35, ε_out=3.2, kmax=50)
    r = abs(z - c) + EPS  # nm
    if r <= Rin + 1e-15
        # Inside: sum_{k=0}^kmax [(k+1)(ε_in-ε_out) / (ε_out(k+1)+ε_in k)] * (r/R)^2k
        s = 0.0
        R = Rin
        for k in 0:kmax
            num = (k + 1) * (ε_in - ε_out)
            den = ε_out * (k + 1) + ε_in * k
            s += (num/den) * (r/R)^(2k)
        end
        return (e2_4pie0_meVnm / (2 * ε_in * R)) * s
    else
        # Outside: sum_{k=1}^kmax [k(ε_out-ε_in) / (ε_out(k+1)+ε_in k)] * (R/r)^{2(k+1)}
        s = 0.0
        R = Rin
        for k in 1:kmax
            num = k * (ε_out - ε_in)
            den = ε_out * (k + 1) + ε_in * k
            s += (num/den) * (R/r)^(2(k+1))
        end
        return (e2_4pie0_meVnm / (2 * ε_out * R)) * s
    end
end

# end self energy potential

function dipole_element(coeffs_m::AbstractVector, coeffs_n::AbstractVector, basis_func, L::Real)
    ψm = QuantumWell.wf_realspace(coeffs_m, basis_func, L)
    ψn = QuantumWell.wf_realspace(coeffs_n, basis_func, L)
    val, _ = quadgk(z -> ψm(z) * z * ψn(z), -L/2, L/2)
    return val
end

# sweep E_field from -5 to 5 in 0.2 mV/nm increments
# eF_sweep = -5:0.01:5
eF_sweep = -4:0.1:4

# build 4 arrays, one for each of the first 4 energies (now 0,1,2,3)
energies_0 = Float64[]
energies_1 = Float64[]
energies_2 = Float64[]
energies_3 = Float64[]

# build 3 arrays, one for each dipole transition (now 1→0, 2→0, 3→0)
dipole_10_array = Float64[]
dipole_20_array = Float64[]
dipole_30_array = Float64[]

const N = 50

for eF in eF_sweep
    V(z) = PotentialWell.potential_well(z) + eF * z + selfenergy(z)

    H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V, L)
    eigvals, eigvecs = eigen(H)
    # Calculate dipole transitions: <1|z|0>, <2|z|0>, <3|z|0>

    # estandarización de los autovectores (?)
    for j in 1:size(eigvecs, 2)
        if eigvecs[1,j] < 0
            eigvecs[:,j] .= -eigvecs[:,j]
        end
    end

    dipole_10 = dipole_element(eigvecs[:,2], eigvecs[:,1], QuantumWell.psi_inf, L)
    dipole_20 = dipole_element(eigvecs[:,3], eigvecs[:,1], QuantumWell.psi_inf, L)
    dipole_30 = dipole_element(eigvecs[:,4], eigvecs[:,1], QuantumWell.psi_inf, L)

    push!(dipole_10_array, dipole_10)
    push!(dipole_20_array, dipole_20)
    push!(dipole_30_array, dipole_30)

    push!(energies_0, eigvals[1])
    push!(energies_1, eigvals[2])
    push!(energies_2, eigvals[3])
    push!(energies_3, eigvals[4])
end

# plot(eF_sweep, energies_0, label="n=0", xlabel="E_field (meV/nm)", ylabel="Energy (meV)", legend=:topright)
# plot!(eF_sweep, energies_1, label="n=1")
# plot!(eF_sweep, energies_2, label="n=2")
# plot!(eF_sweep, energies_3, label="n=3")

p = plot(eF_sweep, abs.(dipole_10_array), label="|⟨1|z|0⟩|",
         xlabel="E_field (mV/nm)", ylabel=L"|M_{ij}| (nm)", legend=:topright)
plot!(p, eF_sweep, abs.(dipole_20_array), label="|⟨2|z|0⟩|")
plot!(p, eF_sweep, abs.(dipole_30_array), label="|⟨3|z|0⟩|")

savefig("SelfEnergyDipoleTransitions.png")
