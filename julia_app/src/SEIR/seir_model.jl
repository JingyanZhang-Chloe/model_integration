# seir_model.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 02/02/2026
=#

using Revise
using Random
include("SEIRModels.jl")
using .Logic
using .Value


function main()
    t = collect(1.0:100.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t, plot=false)
    I_data = I .+ 0.01 .* I .* randn(length(I))
    u0 = [0.00003, 0.03, 0.003, 8000, 2]
    results = Logic.run_experiments(u0, I_data, t)
    Logic.print_results(results)
end

main()