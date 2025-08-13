using Plots

gr()

const nm = 1e-9                 # metros
const meV = 1e-3 * 1.60218e-19  # Joules
const m_e = 9.10938e-31         # kg
const hbar = 1.05457e-34        # Js

const L = 60.0
const Z_1 = -11.0
const Z_2 = -7.0
const Z_3 = 3.0
const Z_4 = 7.0
const Z_5 = 15.0

const dz = 0.1

z = -L/2:dz:L/2

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

plot(z, V.(z), dpi=300)

savefig("pozo.png")