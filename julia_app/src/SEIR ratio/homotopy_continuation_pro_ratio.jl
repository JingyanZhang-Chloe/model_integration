# homotopy_continuation_pro_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

using HomotopyContinuation, DynamicPolynomials, Random

include("SEIRModels_ratio.jl")
using .Logic_R
using .Value_R

@var α, σ, γ, S0, E0
const scaled_variables = [α, σ, γ, S0, E0]

function comp_best_result(t::Vector, I::Vector, I_data::Vector, scaled_vars::Vector, method::String)
    B = Logic_R.get_blocks(I_data, t, method)
    I_hat = Logic_R.comp_I_hat(scaled_vars, B..., t)
    J = sum((I_hat .- I_data).^2) / length(t)
    system_eqs = differentiate(J, scaled_vars)
    C = System(system_eqs, variables=scaled_vars)

    scaled_results = HomotopyContinuation.solve(C)
    real_scaled_results = real_solutions(scaled_results)
    real_results = [result .* Value_R.scales for result in real_scaled_results]

    filtered_results = filter(real_results) do res
        all(Value_R.lb .<= res .<= Value_R.ub) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC. The best solution is selected based on all real solutions"
        filtered_results = real_results
    end

    best_result, best_err = Logic_R.best_solution(filtered_results, I_data, B..., t)
    err = Logic_R.get_error(best_result)
    err_I_data = Logic_R.RSS_I_data(I_data, I)
    println("  Method: $method")
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
    println("  RSS (sum((I_data .- I).^2)): $err_I_data")
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic_R.simulate_seir(t)
    I_data = I .+ 0 .* I .* randn(length(I))
    comp_best_result(t, I, I_data, scaled_variables, "S")
end

main()

