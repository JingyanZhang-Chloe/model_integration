# homotopy_continuation_pro.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 03/02/2026

Major Improvement and Changes:
- Estimation of Double Integrals (TO DO)
- Normalize the Cost Function : J = sum((I_hat .- I_data).^2) / length(I_data)
- Add Bounds Filter
- Rescale
- accuracy=1e-8,
  refinement_accuracy = 1e-12

=#

using Revise
using HomotopyContinuation, DynamicPolynomials, JLD2, Random
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const variables_scaled = [α, σ, γ, S0, E0]

function _comp_I_hat(I_data::Vector, t::Vector, v_s::Vector)::Vector
    B1, B2, B3, B4, B5, B6 = Logic.get_blocks_simpson(I_data, t)
    I0 = I_data[1]

    α_eff  = v_s[1] * Value.scales[1]
    σ_eff  = v_s[2] * Value.scales[2]
    γ_eff  = v_s[3] * Value.scales[3]
    S0_eff = v_s[4] * Value.scales[4]
    E0_eff = v_s[5] * Value.scales[5]

    C1 = σ_eff * (E0_eff - I0) .* t .* B1
    C2 = - (γ_eff + σ_eff) .* B2
    C3 = - 0.5 * α_eff .* B3
    C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* B4
    C5 = - α_eff * (γ_eff + σ_eff) .* B5
    C6 = - 0.5 * α_eff * σ_eff * γ_eff .* B6

    return C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
end

function comp_results(t::Vector, I_data::Vector, vars_scaled::Vector, lb::Vector=Value.lb, ub::Vector=Value.ub, save_name::String="real_solution_homotopy_pro.jld2", save::Bool=false)
    I_hat = _comp_I_hat(I_data, t, vars_scaled)
    J = sum((I_hat .- I_data).^2) / length(I_data)
    system_eqs = differentiate(J, vars_scaled)
    C = System(system_eqs, variables=vars_scaled)

    result_scaled = HomotopyContinuation.solve(C;
        start_system = :polyhedral,
        tracker_options = TrackerOptions(automatic_differentiation=3),
    )

    real_results_scaled = real_solutions(result_scaled)
    real_results = [result .* Value.scales for result in real_results_scaled]

    bounded_results = filter(real_results) do res
        all(lb .<= res .<= ub)
    end

    if save
        path = joinpath(@__DIR__, save_name)
        jldsave(path; bounded_results)
    end

    return bounded_results
end

function print_best_solution(t, noise, vars_scaled)
    S, E, I, R = Logic.simulate_seir(t, plot=false)
    I_data = I .+ noise .* I .* randn(length(I))
    results = comp_results(t, I_data, vars_scaled)
    best_result, best_err = Logic.best_solution(results, I_data, t)
    err = Logic.get_error(best_result)
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
end

function main()
    t = collect(0.0:10.0:1000.0)
    print_best_solution(t, 0.001, variables_scaled)
end

main()