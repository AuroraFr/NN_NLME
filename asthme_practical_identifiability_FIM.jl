using DifferentialEquations, ForwardDiff, LinearAlgebra

# ---------------------------------------------------------------------
# 1. Fixed constants
# ---------------------------------------------------------------------
const ηap   = 1.0
const ηac   = 1.0
const ηapm  = 10.0
const γp    = 1.0
const γc    = 1.0
const γa    = 0.01
const pmax  = 1.0
const κpc = 1.0
const κpm = 0.1
const κcp = 0.01
const κac = 0.01
const φm = 0.01
const κb = 1.0
const φc = 0.1
const k_apm= 0.1
const k_ap = 1.0
const κp = 1.15
# ---------------------------------------------------------------------
# 2. Define the stimulus function
# ---------------------------------------------------------------------
function stimulus_func(t; k_s=0.2, ν=30.0, t_is=[50, 90, 130, 170, 210])
    s = 0.0
    for t_i in t_is
        s += (k_s / ν) * exp(-((t - t_i)^2) / ν^2)
    end
    return s
end

# ---------------------------------------------------------------------
# 3. Define ODE system (stimulus(t) as forcing)
# ---------------------------------------------------------------------
function TGFβ!(du, u, θ, t)
    p_asm, c, a, m = u
    log_kap, log_kb = θ

    stim = stimulus_func(t)  # <-- external input

    du[1] = κp * p_asm * (1 - p_asm/pmax) * (1 + (exp(log_kap) * a * p_asm) / (ηap + a * p_asm)) + κcp*(γc/γp)*c - κpc*p_asm
    du[2] = κpc*(γp/γc)*p_asm - (κcp + φc)*c
    du[3] = (stim + (κac*a*c)/(ηac + a*c)) * c*m/γa - exp(log_kb) * (γp*p_asm/γc + c) * a - a
    du[4] = (κpm + (k_apm * a * p_asm) / (ηapm + a * p_asm)) * γp * p_asm - φm * m
    
end

# ---------------------------------------------------------------------
# 3. Nominal parameter vector (population means)
# ---------------------------------------------------------------------
θ_nom = [log(1), log(1)]   # estimated parameters
u0    = [0.1, 0.8, 0.01, 0.9]         # fixed known ICs
tspan = (0.0, 400.0)

# ---------------------------------------------------------------------
# 4. Solve ODE at nominal values
# ---------------------------------------------------------------------
prob = ODEProblem(TGFβ!, u0, tspan, θ_nom)
sol  = solve(prob, Tsit5(), saveat=0:10:400)

# ---------------------------------------------------------------------
# 5. Compute numerical sensitivity of a(t) (state 3) w.r.t. parameters
# ---------------------------------------------------------------------
times = 0:10:400
S = ForwardDiff.jacobian(p -> (solve(prob, p=p, saveat=times)[3,:]), θ_nom)

# Fisher Information Matrix (approximate practical identifiability)
F = S' * S/0.01
println("Rank(F) = ", rank(F), " of ", length(θ_nom))
println("Condition number of F = ", cond(F))

U, Σ, Vt = svd(F)
println(Σ)  # singular values
println(Vt[end, :])  # the last row gives the nearly-unidentifiable parameter combination

    