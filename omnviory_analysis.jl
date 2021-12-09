
using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve

@with_kw mutable struct Par 
    r = 5.0
    K = 1.5
    h_CR = 0.5
    h_PC = 0.5
    h_PR = 0.5
    e_CR = 0.5
    e_PC = 0.5
    e_PR = 0.5
    m_C = 0.5
    m_P = 0.5
    a_CR = 3.0
    a_PR = 0.5
    
end

function model!(du, u, p, t)
    @unpack r, K, e_CR, e_PC, e_PR,  a_CR, a_PR, h_CR, h_PC, h_PR, m_C, m_P = p 
    
    a_PC = 2.0 - a_PR 

    R, C, P = u
    
    du[1]= r * R * (1 - ( R)/K) - (a_CR * R * C)/(1 + a_CR * h_CR * R) - (a_PR * R * P)/(1 + a_PR * h_PR * R + a_PC * h_PC * C)
    

    du[2] = (e_CR * a_CR * R * C)/(1 + a_CR * h_CR * R) - (a_PC * C * P)/(1 + a_PR * h_PR * R + a_PC * h_PC * C) - m_C * C
    
    
    du[3] = (e_PR * a_PR * R * P + e_PC * a_PC * C * P)/(1 + a_PR * h_PR * R + a_PC * h_PC * C) - m_P * P

    return 
end

## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function model(u, Par, t)
    du = similar(u)
    model!(du, u, Par, t)
    return du
end

## calculating the jacobian 
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end



## plot time series 
let
    u0 = [0.5, 0.3, 0.3]
    t_span = (0, 10000.0)
    p = Par(a_PR=1.45)
    ts =range(9500, 10000, length = 500)
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    model_ts = figure()
    PyPlot.plot(sol.t, sol.u)
    xlabel("Time")
    ylabel("Density")
    legend(["R", "C", "P"])
    return model_ts

end


## equilibrium check for parameter range -- find where all species coexist (interior equilibrium)


om_vals = 0.0:0.05:1.5
om_hold = fill(0.0,length(om_vals),4)

u0 = [0.5, 0.3, 0.3]
t_span = (0, 10000.0)
p = Par()
ts =range(9500, 10000, length = 500)
prob = ODEProblem(model!, u0, t_span, p)
sol = solve(prob, reltol = 1e-8, atol = 1e-8)
grid = sol(ts)
eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero

for i=1:length(om_vals)
    p = Par(a_PR=om_vals[i])
    if i==1
        u0 = [0.5, 0.3, 0.3]
       
      else 
        u0 = [eq[1], eq[2], eq[3]]
      end 
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    equ = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    om_hold[i,1] = om_vals[i]
    om_hold[i,2:end] = equ
    println(om_hold[i,:])
end


using Plots

plot1 = Plots.plot(om_hold[:,1], om_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Predator Equilibrium Density " )

plot2 = Plots.plot(om_hold[:,1], om_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Consumer Equilibrium Density " )

plot3 = Plots.plot(om_hold[:,1], om_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Resource Equilibrium Density " )


eig_hold = fill(0.0,length(om_vals),4)

for i=1:length(om_vals)
    p = Par(a_PR=om_vals[i])
    if i==1
        u0 = [0.5, 0.3, 0.3]
      else 
        u0 = [equ[1], equ[2], equ[3]]
      end 
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    omnivory_jac = jac(eq, model, p)
    all_eig = real.(eigvals(omnivory_jac))
    eig_hold[i,1] = om_vals[i]
    eig_hold[i,2:end] = all_eig
    println(eig_hold[i,:])
end


plot4 = Plots.plot(eig_hold[:,1], eig_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Eig 1 " )

plot5 = Plots.plot(eig_hold[:,1], eig_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Eig 2 " )

plot6 = Plots.plot(eig_hold[:,1], eig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Eig 3 " )


maxeig_hold = fill(0.0,length(om_vals),2)

for i=1:length(om_vals)
    p = Par(a_PR=om_vals[i])
    if i==1
        u0 = [0.5, 0.3, 0.3]
      else 
        u0 = [equ[1], equ[2], equ[3]]
      end 
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, atol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    omnivory_jac = jac(eq, model, p)
    max_eig = maximum(real.(eigvals(omnivory_jac)))
    maxeig_hold[i,1] = om_vals[i]
    maxeig_hold[i,2] = max_eig
    println(maxeig_hold[i,:])
end

plot7 = Plots.plot(maxeig_hold[:,1], maxeig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " Strength of Omnivory ", ylabel = " Real Max Eig " )
