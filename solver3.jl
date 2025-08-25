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

function dipole_element(coeffs_m::AbstractVector, coeffs_n::AbstractVector, basis_func, L::Real)
    ψm = QuantumWell.wf_realspace(coeffs_m, basis_func, L)
    ψn = QuantumWell.wf_realspace(coeffs_n, basis_func, L)
    val, _ = quadgk(z -> ψm(z) * z * ψn(z), -L/2, L/2)
    return val
end

# sweep E_field from -5 to 5 in 0.2 meV/nm increments
eF_sweep = -5:0.01:5

# build 4 arrays, one for each of the first 4 energies
energies_1 = Float64[]
energies_2 = Float64[]
energies_3 = Float64[]
energies_4 = Float64[]

# build 3 arrays, one for each dipole transition
dipole_21_array = Float64[]
dipole_31_array = Float64[]
dipole_41_array = Float64[]

const N = 50

for eF in eF_sweep
    V(z) = PotentialWell.potential_well(z) + eF * z

    H = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V, L)
    eigvals, eigvecs = eigen(H)
    # Calculate dipole transitions: <2|z|1>, <3|z|1>, <4|z|1>

    dipole_21 = dipole_element(eigvecs[:,2], eigvecs[:,1], QuantumWell.psi_inf, L)
    dipole_31 = dipole_element(eigvecs[:,3], eigvecs[:,1], QuantumWell.psi_inf, L)
    dipole_41 = dipole_element(eigvecs[:,4], eigvecs[:,1], QuantumWell.psi_inf, L)

    push!(dipole_21_array, dipole_21)
    push!(dipole_31_array, dipole_31)
    push!(dipole_41_array, dipole_41)

    push!(energies_1, eigvals[1])
    push!(energies_2, eigvals[2])
    push!(energies_3, eigvals[3])
    push!(energies_4, eigvals[4])
end

# plot(eF_sweep, energies_1, label="n=1", xlabel="E_field (meV/nm)", ylabel="Energy (meV)", legend=:topright)
# plot!(eF_sweep, energies_2, label="n=2")
# plot!(eF_sweep, energies_3, label="n=3")
# plot!(eF_sweep, energies_4, label="n=4")

# savefig("quantum_well_energies_vs_E_field.png") 

plot(eF_sweep, dipole_21_array, label="⟨2|z|1⟩", xlabel="E_field (meV/nm)", ylabel="Dipole Transition (meV/nm)", legend=:topright)
plot!(eF_sweep, dipole_31_array, label="⟨3|z|1⟩")
plot!(eF_sweep, dipole_41_array, label="⟨4|z|1⟩")

savefig("quantum_well_dipole_transitions_vs_E_field.png")