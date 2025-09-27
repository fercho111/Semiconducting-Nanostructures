include("helpers/Constants.jl")
include("helpers/QuantumWell.jl")
include("helpers/PotentialWell.jl")

using HDF5
using Plots
using QuadGK
using Unitful
using LinearAlgebra

using .QuantumWell
using .PotentialWell
using .Constants

const L = 60
const N = 200

# --- Coulomb prefactor in meV·nm ---
const e2_4pie0_Jm = Constants.electron^2 / (4 * π * Constants.epsilon_0)  # J·m
const e2_4pie0_meVnm = (e2_4pie0_Jm / Constants.electron) * 1e3 * 1e9     # meV·nm

# modified potential 

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

# end modified potential

# --- Potentials ---
const eF = -4  # 1 meV/nm = 10 kV/cm
V_self_E(z) = PotentialWell.potential_well(z) + selfenergy(z) + eF * z

# --- Solve Hamiltonians ---

H_selfE = QuantumWell.build_hamiltonian(N, QuantumWell.psi_inf, QuantumWell.E_inf, V_self_E, L)
eigvals_selfE, eigvecs_selfE = eigen(H_selfE)

println("With self-energy + E-field energies (meV): ", eigvals_selfE[1:2])

# --- Grid for plotting ---
z_grid = -L/2:0.06:L/2

# --- Plot the potentials ---
plot(z_grid, V_self_E.(z_grid), label="Well + self-energy + E-field", lw=2, ylims=(-200, 300))

# --- Overlay first 2 states for all three cases ---
scale_factor = 500  # arbitrary scaling for visibility

for k in 1:2
    ψ_selfE = QuantumWell.wf_realspace(eigvecs_selfE[:,k], QuantumWell.psi_inf, L)

    ψ_selfE_vals = abs2.(ψ_selfE.(z_grid))

    plot!(z_grid, scale_factor*ψ_selfE_vals .+ eigvals_selfE[k], 
          label="Self+E n=$k", lw=2)
end

xlabel!("z (nm)")
ylabel!("Energy (meV)")
title!("Bare vs. Self-energy vs. Self+E-field (F=$(eF * 10) kV/cm)")


savefig("test.png")