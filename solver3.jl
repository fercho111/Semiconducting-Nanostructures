include("solver2.jl")

# sweep E_field from -5 to 5 in 0.2 increments
E_fields = -5:0.2:5

# build 4 arrays, one for each of the first 4 energies
energies_1 = Float64[]
energies_2 = Float64[]
energies_3 = Float64[]
energies_4 = Float64[]

# build 3 arrays, one for each dipole transition
dipole_21_array = Float64[]
dipole_31_array = Float64[]
dipole_41_array = Float64[]

# Reconstruct eigenfunction as a function of z
function ψ_eig_func(n, eigenvecs,)
    coeffs = myeigvecs[:, n]
    return z -> sum(coeffs[k] * psi_inf(k, z) for k in 1:N)
end


for E in E_fields
    myH = build_hamiltonian_E(N, psi_inf, E_inf, potential_well; E_field=E, q=1.0)
    myeigvals, myeigvecs = eigen(myH)
    # calculate dipole transition for the first 3 moments
    # Calculate dipole transitions: <2|z|1>, <3|z|1>, <4|z|1>
    
    psi1 = myeigvecs[:, 1] / norm(myeigvecs[:, 1])
    psi2 = myeigvecs[:, 2] / norm(myeigvecs[:, 2])
    psi3 = myeigvecs[:, 3] / norm(myeigvecs[:, 3])
    psi4 = myeigvecs[:, 4] / norm(myeigvecs[:, 4])

    # Reconstruct eigenfunction as a callable function for quadgk
    ψ_eig_func(n) = z -> sum(myeigvecs[:, n] .* [psi_inf(k, z) for k in 1:N])

    dipole_21, _ = quadgk(z -> ψ_eig_func(2)(z) * z * ψ_eig_func(1)(z), -L/2, L/2)
    dipole_31, _ = quadgk(z -> ψ_eig_func(3)(z) * z * ψ_eig_func(1)(z), -L/2, L/2)
    dipole_41, _ = quadgk(z -> ψ_eig_func(4)(z) * z * ψ_eig_func(1)(z), -L/2, L/2)

    push!(dipole_21_array, dipole_21)
    push!(dipole_31_array, dipole_31)
    push!(dipole_41_array, dipole_41)

    push!(energies_1, myeigvals[1])
    push!(energies_2, myeigvals[2])
    push!(energies_3, myeigvals[3])
    push!(energies_4, myeigvals[4])
end

# plot energies as a function of E_field
using Plots

# plot(E_fields, energies_1, label="n=1", xlabel="E_field (meV/nm)", ylabel="Energy (meV)", legend=:topright)
# plot!(E_fields, energies_2, label="n=2")
# plot!(E_fields, energies_3, label="n=3")
# plot!(E_fields, energies_4, label="n=4")
# 
# savefig("quantum_well_energies_vs_E_field.png")

# plot dipole transitions as a function of E_field
using Plots

plot(E_fields, dipole_21_array, label="⟨2|z|1⟩", xlabel="E_field (meV/nm)", ylabel="Dipole Transition (meV/nm)", legend=:topright)
plot!(E_fields, dipole_31_array, label="⟨3|z|1⟩")
plot!(E_fields, dipole_41_array, label="⟨4|z|1⟩")

savefig("quantum_well_dipole_transitions_vs_E_field.png")