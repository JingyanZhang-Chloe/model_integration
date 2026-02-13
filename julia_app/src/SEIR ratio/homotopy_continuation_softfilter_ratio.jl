# homotopy_continuation_softfilter_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 13/02/2026
=#

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
const variables = [α, σ, γ, S0, E0]

function project_to_bounds(result::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})
    x = copy(result)
    for i in eachindex(x)
        x[i] = max(x[i], lb[i])
        x[i] = min(x[i], ub[i])
    end

    if x[4] + x[5] > 1
        # Reduce res[5], but make sure it is above the lower bound
        x[5] = max(1 - x[4], lb[5])

        if x[4] + x[5] > 1
            # If after lowering res[5] we are still out of bounds, lets reduce res[4]
            x[4] = max(1 - x[5], lb[4])
        end
    end

    return x
end


function comp_best_result(t::Vector, I::Vector, I_data::Vector, vars::Vector, method::String)
    B = Logic_R.get_blocks(I_data, t, method)
    I_hat = Logic_R.residual(vars, B..., t)
    J = sum((I_hat .- I_data).^2) / length(t)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    results = HomotopyContinuation.solve(C)
    real_results = real_solutions(results)

    lb = Value_R.lb
    ub = Value_R.ub

    filtered_results = filter(real_results) do res
        all(lb .<= res .<= ub) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
    end

    tol = 1e-4
    if isempty(filtered_results)
        @error "No physical solutions found by hard filter. Try soften the filter. tolerance $tol."
        filtered_results = filter(real_results) do res
            ok_bounds =
            all(res .>= lb .- tol) &&
            all(res .<= ub .+ tol)

            ok_physics =
            (res[2] > res[3]) &&
            (res[4] + res[5] <= 1 + tol)

            ok_bounds && ok_physics
        end
    end

    if isempty(filtered_results)
        @error "No physical solutions found by hard ADN soft filter. The best solution is selected based on all real solutions"
        filtered_results = real_results
    else
        println("Found physical solutions with soft filter. Mapping back to bounds")
        filtered_results_bounds = []
        for r in filtered_results
            append!(filtered_results_bounds, project_to_bounds(r, lb, ub))
        end
        filtered_results = filtered_results_bounds
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
    I_data = I .+ 0.0001 .* I .* randn(length(I))
    comp_best_result(t, I, I_data, variables, "S")
end

main()


