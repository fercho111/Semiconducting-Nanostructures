# modify the potential with an e F x term (applied elec)
include("solver.jl")

const E_field = 4 # meV/m


function build_hamiltonian_E(N, ψ, E_kin, V; E_field=4, q=1.0)
    H = zeros(N, N)
    for m in 1:N
        for n in 1:N
            if m == n
                # Kinetic part (diagonal in infinite well basis)
                H[m, n] += E_kin(n)
            end
            # Total potential: original potential + electric field term
            total_potential(z) = V(z) + E_field * z  # Linear field term

            # Potential matrix element: ⟨m|V|n⟩
            val, _ = quadgk(z -> ψ(m, z) * total_potential(z) * ψ(n, z), -L/2, L/2)
            H[m, n] += val
        end
    end
    return H
end


# start from here

# test
# N = 50
# H = build_hamiltonian_E(N, psi_inf, E_inf, potential_well; E_field=E_field, q=1.0)
# 
# eigvals, eigvecs = eigen(H)
# 
# println("First 5 energies (meV):")
# println(eigvals[1:5])
# 
# z_grid = range(-L/2, L/2, length=1000)
# 
# function wf_realspace(coeffs, basis_func, zgrid)
#     # Sum over basis functions with the given coefficients
#     [sum(coeffs[n] * basis_func(n, z) for n in 1:length(coeffs)) for z in zgrid]
# end
# 
# # Convert energies to meV
# energies = eigvals
# 
# # Potential over the same grid, plus the electric field term
# V_grid = [potential_well(z) + E_field * z for z in z_grid]
# 
# # Plot potential first
# plot(z_grid, V_grid, label="Potential", lw=2, color=:black)
# 
# # Overlay first 4 states
# scale_factor = 500  # arbitrary vertical scaling for visibility
# for k in 1:4
#     ψ_k = abs2.(wf_realspace(eigvecs[:,k], psi_inf, z_grid))
#     plot!(z_grid, scale_factor * ψ_k .+ energies[k], label="n=$k")
# end
# 
# xlabel!("z (nm)")
# ylabel!("Energy (meV)")
# 
# savefig("quantum_well_states_E_field.png")
# 
# 