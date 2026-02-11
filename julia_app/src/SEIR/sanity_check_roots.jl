# sanity_check_roots.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 05/02/2026
=#

using HomotopyContinuation
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const vars_scaled = [α, σ, γ, S0, E0]

function roots_check(t::Vector, I_data::Vector, vars_scaled::Vector, method::String; err_tol::Float64=0.01)
    B = Logic.get_blocks(I_data, t, method)
    I_hat = Logic.comp_I_hat(vars_scaled, B..., t)
    J = sum((I_hat .- I_data).^2) / length(I_data)
    system_eqs = differentiate(J, vars_scaled)
    C = System(system_eqs, variables=vars_scaled)

    result_scaled = HomotopyContinuation.solve(C)
    real_results = [result .* Value.scales for result in real_solutions(result_scaled)]
    filtered_results = filter(real_results) do res
        return all(Value.lb .<= res .<= Value.ub) && (res[2] > res[3])
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC."
    end

    filtered_results_scaled = [result ./ Value.scales for result in filtered_results]

    println("===== For all bounded results =====")
    for (i, res_scaled) in enumerate(filtered_results_scaled)
        y = evaluate(C, res_scaled)
        err_sq = sum(abs.(y))
        I_hat_num = Logic.comp_I_hat(res_scaled, B..., t)
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
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    I_data = I .+ 0 .* I .* randn(length(I))
    roots_check(t, I_data, vars_scaled, "T")
end

main()
