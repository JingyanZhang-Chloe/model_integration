# hc_ideal_integral.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 09/02/2026

Major Improvement and Changes:
- Estimation of Double Integrals (TO DO)
- Normalize the Cost Function : J = sum((I_hat .- I_data).^2) / length(I_data)
- Add Bounds Filter
- Rescale
- solve
    start_system = :polyhedral,
    tracker_options = TrackerOptions(automatic_differentiation=3)

In this file we use integrals from extending the ODEs to measure the gain from ideal integrals. No noise applied
=#

using HomotopyContinuation, DifferentialEquations
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const variables_scaled = [α, σ, γ, S0, E0]

function _comp_ideal_I_hat_check(t, v_s, I0, B1, B2, B3, B4, B5, B6, I_true)
    α_eff  = v_s[1] * Value.scales[1]
    σ_eff  = v_s[2] * Value.scales[2]
    γ_eff  = v_s[3] * Value.scales[3]
    S0_eff = v_s[4] * Value.scales[4]
    E0_eff = v_s[5] * Value.scales[5]

    C1 = σ_eff * (E0_eff + I0) .* t .* B1
    C2 = - (γ_eff + σ_eff) .* B2
    C3 = - 0.5 * α_eff .* B3
    C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* B4
    C5 = - α_eff * (γ_eff + σ_eff) .* B5
    C6 = - 0.5 * α_eff * σ_eff * γ_eff .* B6

    I_hat = I0 .+ C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6

    println("=== DIAGNOSTIC REPORT ===")
    println("True I (end): $(I_true[end])")
    println("Est I  (end): $(I_hat[end])")
    println("Difference:   $(I_hat[end] - I_true[end])")
    println("-------------------------")
    println("Term Contributions at final time T:")
    println("I0: $I0")
    println("C1 (Linear):  $(C1[end])")
    println("C2 (SingInt): $(C2[end])")
    println("C3 (SqInt):   $(C3[end])")
    println("C4 (DblInt):  $(C4[end])")
    println("C5 (DblSq):   $(C5[end])")
    println("C6 (NestSq):  $(C6[end])")
    println("Sum:          $(I0 + C1[end] + C2[end] + C3[end] + C4[end] + C5[end] + C6[end])")
    println("=========================")

    return I_hat
end

function comp_ideal_results(t::Vector{Float64}, vars_scaled::Vector, lb::Vector=Value.lb, ub::Vector=Value.ub, save_name::String="real_solution_homotopy_pro.jld2", save::Bool=false)
    I0, B1, B2, B3, B4, B5, B6, I_consistent = Logic.get_ideal_blocks(t)
    I_hat = Logic.comp_I_hat(vars_scaled, I0, B1, B2, B3, B4, B5, B6, t)
    J = sum((I_hat .- I_consistent).^2) / length(I_consistent)
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

    return bounded_results, I_consistent
end

function print_ideal_best_solution(t, vars_scaled)
    results, I_consistent = comp_ideal_results(t, vars_scaled)
    best_result, best_err = Logic.best_solution(results, I_consistent, t)
    err = Logic.get_error(best_result)
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_consistent).^2)): $best_err")
end

function sanity_check(t)
    I0, B1, B2, B3, B4, B5, B6, I_consistent = get_ideal_blocks(t)
    p_true_scaled = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0] ./ Value.scales
    I_hat_manual = _comp_ideal_I_hat(t, p_true_scaled, I0, B1, B2, B3, B4, B5, B6)
    rss_manual = sum((I_hat_manual .- I_consistent).^2)

    println("========================================")
    println("MANUAL SANITY CHECK (True Parameters):")
    println("RSS: $rss_manual")
    println("========================================")
end

function main()
    t = collect(0.0:10.0:1000.0)
    p_true_scaled = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0] ./ Value.scales
    print_ideal_best_solution(t, variables_scaled)
    # sanity_check(t)
    # I0, B1, B2, B3, B4, B5, B6, I_consistent = get_ideal_blocks(t)
    # _comp_ideal_I_hat_check(t, p_true_scaled, I0, B1, B2, B3, B4, B5, B6, I_consistent)
end

main()
