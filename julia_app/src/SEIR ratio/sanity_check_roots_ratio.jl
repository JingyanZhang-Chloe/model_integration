# sanity_check_roots_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

using HomotopyContinuation
include("SEIRModels_ratio.jl")
using .Value_R
using .Logic_R

@var α, σ, γ, S0, E0
const variables = [α, σ, γ, S0, E0]

function roots_check(t::Vector, I_data::Vector, vars::Vector, method::String; err_tol::Float64=0.01)
    B = Logic_R.get_blocks(I_data, t, method)
    I_hat = Logic_R.comp_I_hat(vars, B..., t)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    results = HomotopyContinuation.solve(C)
    real_results = HomotopyContinuation.real_solutions(results)
    filtered_results = filter(real_results) do res
        return all(Value_R.lb .<= res .<= Value_R.ub) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC. Checking all real solutions"
        filtered_results = real_results
    end

    for (i, res) in enumerate(filtered_results)
        y = evaluate(C, res)
        err_sq = sum(abs.(y))
        I_hat_num = Logic_R.comp_I_hat(res, B..., t)
        err_rss = sum((I_hat_num .- I_data).^2)
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
    S, E, I, R = Logic_R.simulate_seir(t)
    I_data = I .+ 0.0001 .* I .* randn(length(I))
    roots_check(t, I_data, variables, "S")
end

main()
