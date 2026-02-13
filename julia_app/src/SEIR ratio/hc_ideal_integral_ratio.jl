# hc_ideal_integral_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

using HomotopyContinuation, DifferentialEquations
include("SEIRModels_ratio.jl")
using .Logic_R
using .Value_R

@var α, σ, γ, S0, E0
const variables = [α, σ, γ, S0, E0]

function comp_ideal_results(t::Vector{Float64}, vars::Vector)
    I0, B1, B2, B3, B4, B5, B6, I_consistent = Logic_R.get_ideal_blocks(t)
    I_hat = Logic_R.residual(vars, I0, B1, B2, B3, B4, B5, B6, t)
    J = sum((I_hat .- I_consistent).^2) / length(t)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    results = HomotopyContinuation.solve(C;
        start_system = :polyhedral,
    )

    real_results = HomotopyContinuation.real_solutions(results)
    solutions = HomotopyContinuation.solutions(results)

    tol = 1e-8
    almost_real = [real.(s) for s in solutions if maximum(abs.(imag.(s))) < tol]

    filtered_results = filter(real_results) do res
        all(Value_R.lb .<= res .<= Value_R.ub) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
    end

    if isempty(filtered_results)
        @error "No real physical solutions found by HC. Searching almost real solutions..."
        filtered_results = filter(almost_real) do res
            all(Value_R.lb .<= res .<= Value_R.ub) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
        end
    end

    if isempty(filtered_results)
        @error "No real or almost real physical solutions found by HC."
        return
    end

    best_result, best_err = Logic_R.best_solution(filtered_results, I_consistent, I0, B1, B2, B3, B4, B5, B6, t)

    err = Logic_R.get_error(best_result)
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_consistent).^2)): $best_err")
end

function sanity_check(t)
    I0, B1, B2, B3, B4, B5, B6, I_consistent = Logic_R.get_ideal_blocks(t)
    p_true = [Value_R.α, Value_R.σ, Value_R.γ, Value_R.S0, Value_R.E0]
    I_hat_manual = Logic_R.residual(p_true, I0, B1, B2, B3, B4, B5, B6, t)
    rss_manual = sum((I_hat_manual .- I_consistent).^2)

    println("========================================")
    println("MANUAL SANITY CHECK (True Parameters):")
    println("RSS: $rss_manual")
    println("========================================")
end

function main()
    t = collect(0.0:10.0:1000.0)
    comp_ideal_results(t, variables)
end

main()
