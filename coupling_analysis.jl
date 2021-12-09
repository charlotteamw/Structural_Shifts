using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve

pygui(true)

@with_kw mutable struct Par 
    
    r_litt = 1.5
    r_pel = 1.5
    α_pel = 0.0  ##competitive influence of pelagic resource on littoral resource 
    α_litt = 0.0   ## competitve influence of littoral resource on pelagic resource
    K = 1.5
    h_CR = 0.8
    h_PC = 0.6
    h_PR = 0.9
    e_CR = 0.7
    e_PC = 0.6
    e_PR = 0.4
    m_C = 0.3
    m_P = 0.3
    a_CR_litt = 1.0
    a_CR_pel = 1.0
    a_PR_litt = 0
    a_PR_pel = 0
    a_PC_litt = 1.0
    a_PC_pel = 1.0
    Σ = 0.0
    
    
end

function coupling_model!(du, u, p, t)

    @unpack r_litt, r_pel, K, α_pel, α_litt, e_CR, e_PC, e_PR,  a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, h_CR, h_PC, h_PR, m_C, m_P, a_PC_litt, a_PC_pel, Σ = p 
    
    
    R_l, R_p, C_l, C_p, P = u

    Ω = (Σ * C_p)/(Σ * C_p + (1-Σ)*C_l)
    
    du[1]= r_litt * R_l * (1 - (α_pel * R_p + R_l)/K) - (a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR * R_l) - (a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt*(1-Ω) * h_PC * C_l + a_PC_pel * Ω * h_PC * C_p)
    
    du[2] = r_pel * R_p * (1 - (α_litt * R_l + R_p)/K) - (a_CR_pel * R_p * C_p)/(1 + a_CR_pel * h_CR * R_p) - (a_PR_pel * R_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt* (1-Ω) * h_PC * C_l + a_PC_pel* Ω  * h_PC * C_p)

    du[3] = (e_CR * a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR * R_l) - (a_PC_litt* (1-Ω)  * C_l * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt* (1-Ω)  * h_PC * C_l + a_PC_pel* Ω  * h_PC * C_p) - m_C * C_l
    
    du[4] = (e_CR * a_CR_pel * R_p * C_p)/(1 + a_PC_pel* Ω  * h_PC * R_p) - (a_PC_pel* Ω  * C_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt* (1-Ω)  * h_PC * C_l + a_PC_pel* Ω  * h_PC * C_p) - m_C * C_p
    
    du[5] = (e_PR * a_PR_litt * R_l * P + e_PR * a_PR_pel * R_p * P + e_PC * a_PC_litt * (1-Ω)  * C_l * P + e_PC * a_PC_pel* Ω  * C_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt* (1-Ω)  * h_PC * C_l + a_PC_pel * Ω  * h_PC * C_p) - m_P * P

    return 
end

## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function coupling_model(u, Par, t)
    du = similar(u)
    coupling_model!(du, u, Par, t)
    return du
end

## calculating the jacobian 
function jac(u, coupling_model, p)
    ForwardDiff.jacobian(u -> coupling_model(u, p, NaN), u)
end



## plot time series 
let
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    t_span = (0, 2000.0)
    ts =range(9500, 10000, length = 500)
    p = Par(Σ = 0.0)
    prob = ODEProblem(coupling_model!, u0, t_span, p)
    sol = solve(prob)
    grid = sol(ts)
    model_ts = figure()
    PyPlot.plot(sol.t, sol.u)
    xlabel("Time")
    ylabel("Density")
    legend(["R_litt", "R_pel", "C_litt","C_pel", "P"])
    return model_ts

end


## equilibrium check for parameter range -- find where all species coexist (interior equilibrium)


coup_vals = 0.0:0.005:0.74
coup_hold = fill(0.0,length(coup_vals),6)

u0 = [0.5,0.5, 0.3, 0.3, 0.3]
t_span = (0, 10000.0)
p = Par(Σ = 0.0)
ts =range(9500, 10000, length = 500)
prob = ODEProblem(coupling_model!, u0, t_span, p)
sol = solve(prob, reltol = 1e-8, atol = 1e-8)
grid = sol(ts)
eq = nlsolve((du, u) -> coupling_model!(du, u, p, 0.0), grid.u[end]).zero

for i=1:length(coup_vals)
    p = Par(Σ =coup_vals[i])
    if i==1
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
       
      else 
        u0 = [eq[1], eq[2], eq[3], eq[4], eq[5]]
      end 
    prob = ODEProblem(coupling_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    equ = nlsolve((du, u) -> coupling_model!(du, u, p, 0.0), grid.u[end]).zero
    coup_hold[i,1] = coup_vals[i]
    coup_hold[i,2:end] = equ
    println(coup_hold[i,:])
end

using Plots

eq_R_litt = Plots.plot(coup_hold[:,1], coup_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " R_litt Equilibrium Density " )

eq_R_pel = Plots.plot(coup_hold[:,1], coup_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " R_pel Equilibrium Density " )

eq_C_litt = Plots.plot(coup_hold[:,1], coup_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " C_litt Equilibrium Density " )

eq_C_pel = Plots.plot(coup_hold[:,1], coup_hold[:,5], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " C_pel Equilibrium Density " )

eq_P = Plots.plot(coup_hold[:,1], coup_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " P Equilibrium Density " )


eig_hold = fill(0.0,length(coup_vals),6)

for i=1:length(coup_vals)
    p = Par(a_PC_litt=coup_vals[i])
    if i==1
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
       
      else 
        u0 = [eq[1], eq[2], eq[3], eq[4], eq[5]]
      end 
    prob = ODEProblem(coupling_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    equ = nlsolve((du, u) -> coupling_model!(du, u, p, 0.0), grid.u[end]).zero
    coup_hold[i,1] = coup_vals[i]
    coup_hold[i,2:end] = equ
    coupling_jac = jac(equ, coupling_model, p)
    all_eig = real.(eigvals(coupling_jac))
    eig_hold[i,1] = coup_vals[i]
    eig_hold[i,2:end] = all_eig
    println(eig_hold[i,:])
end


eig_1 = Plots.plot(eig_hold[:,1], eig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 1 " )

eig_2 = Plots.plot(eig_hold[:,1], eig_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 2 " )

eig_3 = Plots.plot(eig_hold[:,1], eig_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 3 " )

eig_4 = Plots.plot(eig_hold[:,1], eig_hold[:,5], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 4 " )

eig_5 = Plots.plot(eig_hold[:,1], eig_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 5 " )

maxeig_hold = fill(0.0,length(coup_vals),2)

for i=1:length(coup_vals)
    p = Par(a_PC_litt = coup_vals[i])
    if i==1
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
       
      else 
        u0 = [eq[1], eq[2], eq[3], eq[4], eq[5]]
      end 
    prob = ODEProblem(coupling_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    equ = nlsolve((du, u) -> coupling_model!(du, u, p, 0.0), grid.u[end]).zero
    coupling_jac = jac(equ, coupling_model, p)
    max_eig = maximum(real.(eigvals(coupling_jac)))
    maxeig_hold[i,1] = coup_vals[i]
    maxeig_hold[i,2] = max_eig
    println(maxeig_hold[i,:])
end


max_eig = Plots.plot(maxeig_hold[:,1], maxeig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Real Max Eig " )
 