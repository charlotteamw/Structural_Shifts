using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


@with_kw mutable struct CoupPar

    ## resource parameters
    r = 1.8
    K = 1.0

    ## consumer parameters
    h_CR = 0.6
    e_CR = 0.7
    m_C = 0.3
    a_CR = 1.2

    ## predator parameters
    h_PC = 0.6
    e_PC = 0.7
    m_P = 0.3
    a_PC = 1.2

    ## predator preference 
    Σ = 0.0
end

function coup_model!(du, u, p, t)

        @unpack  r, K, h_CR, e_CR, m_C, a_CR, h_PC, e_PC, m_P, a_PC, Σ = p 

        
        R1, R2, C1, C2, P = u
    
        Ω = (Σ * C2)/(Σ * C2 + (1-Σ) * C1)
        
        du[1] = r * R1 * (1 - R1/K) - (a_CR * R1 * C1)/(1 + a_CR * h_CR * R1) 
        
        du[2] = r * R2 * (1 - R2/K) - (a_CR * R2 * C2)/(1 + a_CR * h_CR * R2) 

        du[3] = (e_CR * a_CR * R1 * C1)/(1 + a_CR * h_CR * R1) - (a_PC * (1-Ω) * C1 * P)/(1  + a_PC * (1-Ω) * h_PC * C1 + a_PC * Ω * h_PC * C2) - m_C * C1
        
        du[4] = (e_CR * a_CR * R2 * C2)/(1 + a_CR * h_CR * R2) - (a_PC * Ω * C2 * P)/(1 + a_PC * (1-Ω) * h_PC * C1 + a_PC * Ω * h_PC * C2) - m_C * C2
        
        du[5] = ((e_PC * a_PC * (1-Ω) * C1 * P) + (e_PC * a_PC * Ω * C2 * P))/(1 + a_PC * (1-Ω) * h_PC * C1 + a_PC * Ω * h_PC * C2) - m_P * P
    
        return 
end

function coup_model(u, CoupPar, t)
        du = similar(u)
        coup_model!(du, u, CoupPar, t)
        return du
end
    
## calculating the jacobian 
function jac(u, coup_model, p)
        ForwardDiff.jacobian(u -> coup_model(u, p, NaN), u)
end
    
    
    
## plot time series 
let
        u0 = [ 0.5, 0.5, 0.5, 0.5, 0.5]
        t_span = (0, 2000.0)
        ts = range(1000, 1500, length = 500)
        p = CoupPar(Σ = 1.0)
        prob = ODEProblem(coup_model!, u0, t_span, p)
        sol = solve(prob)
        grid = sol(ts)
        model_ts = figure()
        PyPlot.plot(sol.t, sol.u)
        xlabel("Time")
        ylabel("Density")
        legend(["R1", "R2", "C1","C2", "P"])
        return model_ts
    
end
    
## calculate equilibrium densities - using grid instead of sol to remove transient 
coup_vals = 0.0:0.001:1.0
coup_hold = fill(0.0,length(coup_vals),6)

u0 = [0.5,0.5, 0.3, 0.3, 0.3]
p = CoupPar(Σ = 0.0)
t_span = (0, 10000.0)
ts = range(1000, 1500, length = 500)


for i=1:length(coup_vals)
    p = CoupPar(Σ =coup_vals[i])
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]  
    prob = ODEProblem(coup_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> coup_model!(du, u, p, 0.0), grid.u[end]).zero
    coup_hold[i,1] = coup_vals[i]
    coup_hold[i,2:end] = eq
    println(coup_hold[i,:])
end

## plot equilibrium densities 
using Plots

eq_R1 = Plots.plot(coup_hold[:,1], coup_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " R1 Equilibrium Density " )

eq_R2 = Plots.plot(coup_hold[:,1], coup_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " R2 Equilibrium Density " )

eq_C1 = Plots.plot(coup_hold[:,1], coup_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " C1 Equilibrium Density " )

eq_C2 = Plots.plot(coup_hold[:,1], coup_hold[:,5], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " C2 Equilibrium Density " )

eq_P = Plots.plot(coup_hold[:,1], coup_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " P Equilibrium Density " )



## calculate all five eigs 

eig_hold = fill(0.0,length(coup_vals),6)

for i=1:length(coup_vals)
    p = CoupPar(Σ =coup_vals[i])
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    prob = ODEProblem(coup_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> coup_model!(du, u, p, 0.0), grid.u[end]).zero
    coup_hold[i,1] = coup_vals[i]
    coup_hold[i,2:end] = eq
    coup_jac = jac(eq, coup_model, p)
    all_eig = real.(eigvals(coup_jac))
    eig_hold[i,1] = coup_vals[i]
    eig_hold[i,2:end] = all_eig
    println(eig_hold[i,:])
end

## plot all real eigs
eig_1 = Plots.plot(eig_hold[:,1], eig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 1 " )

eig_2 = Plots.plot(eig_hold[:,1], eig_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 2 " )

eig_3 = Plots.plot(eig_hold[:,1], eig_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 3 " )

eig_4 = Plots.plot(eig_hold[:,1], eig_hold[:,5], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 4 " )

eig_5 = Plots.plot(eig_hold[:,1], eig_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Eig 5 " )


## calculate max real eigs 
maxeig_hold = fill(0.0,length(coup_vals),2)

for i=1:length(coup_vals)
    p = CoupPar(Σ = coup_vals[i])
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    prob = ODEProblem(coup_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> coup_model!(du, u, p, 0.0), grid.u[end]).zero
    coup_jac = jac(eq, coup_model, p)
    max_eig = maximum(real.(eigvals(coup_jac)))
    maxeig_hold[i,1] = coup_vals[i]
    maxeig_hold[i,2] = max_eig
    println(maxeig_hold[i,:])
end

## plot max real eig 
max_eig = Plots.plot(maxeig_hold[:,1], maxeig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Coupling ", ylabel = " Real Max Eig " )
 