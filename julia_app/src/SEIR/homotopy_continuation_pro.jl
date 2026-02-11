# homotopy_continuation_pro.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 03/02/2026

Major Improvement and Changes:
- Normalize the Cost Function : J = sum((I_hat .- I_data).^2) / length(I_data)
- Rescale
- solve
    start_system = :polyhedral,
    tracker_options = TrackerOptions(automatic_differentiation=3)
=#

using Revise
using HomotopyContinuation, DynamicPolynomials, Random
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const variables_scaled = [α, σ, γ, S0, E0]

function comp_results_pro(t::Vector, I_data::Vector, vars_scaled::Vector, method::String)
    B = Logic.get_blocks(I_data, t, method)
    I_hat = Logic.comp_I_hat(vars_scaled, B..., t)
    J = sum((I_hat .- I_data).^2) / length(I_data)
    system_eqs = differentiate(J, vars_scaled)
    C = System(system_eqs, variables=vars_scaled)

    result_scaled = HomotopyContinuation.solve(C;
        start_system = :polyhedral,
        tracker_options = TrackerOptions(automatic_differentiation=3),
    )

    real_results_scaled = real_solutions(result_scaled)
    real_results = [result .* Value.scales for result in real_results_scaled]

    filtered_results = filter(real_results) do res
        all(Value.lb .<= res .<= Value.ub) && (res[2] > res[3])
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC, returning un-filtered real results"
        return real_results
    end

    return filtered_results
end

function comp_best_result_pro(t::Vector, I::Vector, I_data::Vector, vars_scaled::Vector, method::String)
    B = Logic.get_blocks(I_data, t, method)
    I_hat = Logic.comp_I_hat(vars_scaled, B..., t)
    J = sum((I_hat .- I_data).^2) / length(I_data)
    system_eqs = differentiate(J, vars_scaled)
    C = System(system_eqs, variables=vars_scaled)

    result_scaled = HomotopyContinuation.solve(C;
        start_system = :polyhedral,
        tracker_options = TrackerOptions(automatic_differentiation=3),
    )

    real_results = [result .* Value.scales for result in real_solutions(result_scaled)]

    filtered_results = filter(real_results) do res
        all(Value.lb .<= res .<= Value.ub) && (res[2] > res[3])
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC. Returning Nothing"
        return nothing
    end

    best_result, best_err = Logic.best_solution(filtered_results, I_data, B..., t)
    err = Logic.get_error(best_result)
    err_I_data = Logic.RSS_I_data(I_data, I)
    println("  Method: $method")
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
    println("  RSS (sum((I_data .- I).^2)): $err_I_data")
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    I_data = I .+ 0.001 .* I .* randn(length(I))
    comp_best_result_pro(t, I, I_data, variables_scaled, "T")
end

main()