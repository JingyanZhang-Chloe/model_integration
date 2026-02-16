# homotopy_continuation_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

using HomotopyContinuation, DynamicPolynomials, Random, Plots, DifferentialEquations

include("SEIRModels_ratio.jl")
using .Logic_R
using .Value_R

@var αT, σT, γT, S0, E0
const variables = [αT, σT, γT, S0, E0]

function simulate_seir(t_scaled, T::Float64; u0=Value_R.u, p=Value_R.p_true, plot=false)
        p = p .* T
        prob = ODEProblem(Logic_R.seir!, u0, (t_scaled[1], t_scaled[end]), p)
        sol = DifferentialEquations.solve(prob, saveat=t_scaled)
        sol_arr = Array(sol)
        S = sol_arr[1, :]
        E = sol_arr[2, :]
        I = sol_arr[3, :]
        R = sol_arr[4, :]

        if plot
            data_to_plot = hcat(S, E, I, R)
            println("Plotting data of size: ", size(data_to_plot))
            plt = Plots.plot(t_scaled, data_to_plot,
            title = "SEIR Model Results",
            label = ["True S" "True E" "True I" "True R"],
            xlabel = "Time",
            ylabel = "Value",
            lw = 2
            )
            display(plt)
        end

        return S, E, I, R
    end

to_physical(res_scaled, T::Float64) = [res_scaled[1] / T, res_scaled[2] / T, res_scaled[3] / T, res_scaled[4], res_scaled[5]]
to_scaled(res, T::Float64) = [res[1] * T, res[2] * T, res[3] * T, res[4], res[5]]

function comp_best_result(t_scaled::Vector, T::Float64, I::Vector, I_data::Vector, vars::Vector, method::String)
    B = Logic_R.get_blocks(I_data, t_scaled, method)
    I_hat = Logic_R.residual(vars, B..., t_scaled)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C)
    real_results_scaled = real_solutions(result)

    lb_scaled = [Value_R.lb[1]*T, Value_R.lb[2]*T, Value_R.lb[3]*T, Value_R.lb[4], Value_R.lb[5]]
    ub_scaled = [Inf, Inf, Inf, Value_R.ub[4], Value_R.ub[5]]

    filtered_results_scaled = filter(real_results_scaled) do res
        all(lb_scaled .<= res .<= ub_scaled) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
    end

    if isempty(filtered_results_scaled)
        @error "No physical solutions found by HC. We project all real results to bounded results instead"
        filtered_results_scaled = Vector{Float64}[]
        for result in real_results_scaled
            bound_result = Logic_R.project_to_bounds(result, lb_scaled, ub_scaled)
            push!(filtered_results_scaled, bound_result)
        end
    end

    best_result_scaled, best_err = Logic_R.best_solution(filtered_results_scaled, I_data, B..., t_scaled)
    best_result = to_physical(best_result_scaled, T)
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
    T = 100.0
    t_scaled = t ./ T
    S, E, I, R = simulate_seir(t_scaled, T, plot=false)
    noise = 0.05
    I_data = I .+ noise .* I .* randn(length(I))

    println("noise level $noise")
    comp_best_result(t_scaled, T, I, I_data, variables, "T")
    comp_best_result(t_scaled, T, I, I_data, variables, "S")
end

main()
