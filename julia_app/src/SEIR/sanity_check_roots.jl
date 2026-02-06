# sanity_check_roots.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 05/02/2026
=#

include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const vars_scaled = [α, σ, γ, S0, E0]

function _comp_I_hat(I_data::Vector, t::Vector, v_s::Vector)::Vector
    B1, B2, B3, B4, B5, B6 = Logic.get_blocks(I_data, t)
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

function roots_check(t::Vector, I_data::Vector, vars_scaled::Vector, lb::Vector, ub::Vector, err_tol::Float64=0.01)
    I_hat = _comp_I_hat(I_data, t, vars_scaled)
    J = sum((I_hat .- I_data).^2) / length(I_data)
    system_eqs = differentiate(J, vars_scaled)
    C = System(system_eqs, variables=vars_scaled)

    result_scaled = HomotopyContinuation.solve(C)
    real_results_scaled = real_solutions(result_scaled)
    bounded_results_scaled = filter(real_results_scaled) do res
        all(lb .<= res .<= ub)
    end
    bounded_results = [result .* Value.scales for result in bounded_results_scaled]

    println("===== For all bounded results =====")
    for (i, res_scaled) in enumerate(bounded_results_scaled)
        y = evaluate(C, res_scaled)
        err_sq = sum(abs.(y))
        I_hat_num = _comp_I_hat(I_data, t, res_scaled)
        err_rss = sum((I_hat_num .- I_data).^2)
        res = res_scaled .* Value.scales
        println("Solution $i:")
        println("  Variables: $res")
        println("  Residual Norm (sum(abs.(evaluate(C, res)))): $err_sq")
        println("  RSS (sum((I_hat .- I_data).^2)): $err_rss")

        if err_sq < err_tol
            println("Valid Root")
        else
            println("Likely not a root")
        end
    end

    println("===== For the best solution selected by min {sum((I_hat .- I_data).^2)} =====")

    best_result, best_err = Logic.best_solution(bounded_results, I_data, t)
    best_result_scaled = best_result ./ Value.scales
    I_hat_num = _comp_I_hat(I_data, t, best_result_scaled)
    err_rss = sum((I_hat_num .- I_data).^2)

    y = evaluate(C, best_result_scaled)
    err_sq = sum(abs.(y))

    @assert err_rss ≈ best_err "RSS calculation mismatch :("
    println("  Variables: $best_result")
    println("  Residual Norm (sum(abs.(evaluate(C, res)))): $err_sq")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")

    if err_sq < err_tol
        println("Valid Root")
    else
        println("Likely not a root")
    end
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    I_data = I .+ 0.001 .* I .* randn(length(I))

    roots_check(t, I_data, vars_scaled, Value.lb, Value.ub)
end

main()
