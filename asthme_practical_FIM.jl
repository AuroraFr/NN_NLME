using DifferentialEquations, ForwardDiff, LinearAlgebra, Statistics, DataFrames

# --- constants (yours) ---
const ηap=1.0; const ηac=1.0; const ηapm=10.0
const γp=1.0; const γc=1.0; const γa=0.01
const pmax=1.0; const κpc=1.0; const κcp=0.01
const κac=0.01; const φm=0.01; const φc=0.1
const k_apm=0.1; const κp=1.15

function stimulus_func(t; k_s=0.2, ν=20.0, t_is=[50,90,130,170,210])
    s = 0.0
    @inbounds for ti in t_is
        s += (k_s/ν)*exp(-((t-ti)^2)/ν^2)
    end
    return s
end

# θ = [log(k_ap), log(k_b)]
function TGFβ_kap_kb!(du,u,θ,t)
    p_asm, c, a, m = u
    log_kap, log_kb = θ
    k_ap = exp(log_kap)
    k_b  = exp(log_kb)
    stim = stimulus_func(t; k_s=0.2)  # keep k_s fixed for this test

    du[1] = κp * p_asm * (1 - p_asm/pmax) * (1 + (k_ap * a * p_asm) / (ηap + a * p_asm)) +
            κcp*(γc/γp)*c - κpc*p_asm
    du[2] = κpc*(γp/γc)*p_asm - (κcp + φc)*c
    du[3] = (stim + (κac*a*c)/(ηac + a*c)) * c*m/γa - k_b * (γp*p_asm/γc + c) * a - a
    du[4] = (0.1 + (k_apm * a * p_asm) / (ηapm + a * p_asm)) * γp * p_asm - φm * m
end

# stack selected states over time; sel=(p,c,a,m)
function simulate_states(θ; u0=[0.1,0.8,0.01,0.9], tspan=(0.0,400.0),
                         saveat=collect(0.0:1:400.0), sel=(false,false,true,false))
    prob = ODEProblem(TGFβ_kap_kb!, u0, tspan, θ)
    sol  = solve(prob, Tsit5(); saveat=saveat, reltol=1e-8, abstol=1e-10)
    Y = Array(sol)  # nstate x nt
    idx = findall(identity, sel)
    vec(reduce(vcat, eachrow(Y[idx, :])))
end

function sensitivities(θ; kwargs...)
    g(ϑ) = simulate_states(ϑ; kwargs...)
    ForwardDiff.jacobian(g, θ)  # nobs x 2
end

function fim_and_diag(θ; σ=0.05, kwargs...)
    S = sensitivities(θ; kwargs...)
    W = Diagonal(fill(1/σ^2, size(S,1)))
    F = S' * W * S
    ρ = dot(S[:,1],S[:,2])/(norm(S[:,1])*norm(S[:,2]))  # column correlation
    U,Σ,Vt = svd(F)
    (; F, cond=cond(F), rank=rank(F), ρ=ρ, svals=Σ, vweak=Vt[end,:])
end

θ0 = [log(1.0), log(1.0)]   # [log k_ap, log k_b]

# (1) a-only, sparse (usually ill-conditioned)
diag_a  = fim_and_diag(θ0; saveat=collect(0.0:1.0:400.0), sel=(false,false,true,false))
println("\n[a only] rank=$(diag_a.rank), cond=$(diag_a.cond), ρ=$(diag_a.ρ)")
println("svals = $(diag_a.svals), weak dir (log[k_ap],log[k_b]) = $(diag_a.vweak)")

# (2) a + p, with dense early window to pin k_b
save_dense = vcat(collect(48.0:1.5:70.0), collect(70.0:1.0:400.0))
diag_ap = fim_and_diag(θ0; saveat=save_dense, sel=(true,false,true,false))
println("\n[a + p, dense early] rank=$(diag_ap.rank), cond=$(diag_ap.cond), ρ=$(diag_ap.ρ)")
println("svals = $(diag_ap.svals), weak dir (log[k_ap],log[k_b]) = $(diag_ap.vweak)")
