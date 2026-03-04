# homotopy_continuation_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

using HomotopyContinuation, DynamicPolynomials, Random, Plots, DifferentialEquations, LsqFit, Statistics

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


function _comp_best_result(t_scaled::Vector, T::Float64, I::Vector, I_data::Vector, vars::Vector, method::String)
    B = Logic_R.get_blocks(I_data, t_scaled, method)
    I_hat = Logic_R.residual(vars, B..., t_scaled)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C)
    real_results_scaled = real_solutions(result)

    lb_scaled = [Value_R.lb[1]*T, Value_R.lb[2]*T, Value_R.lb[3]*T, Value_R.lb[4], Value_R.lb[5]]
    ub_scaled = [Inf, Inf, Inf, Value_R.ub[4], Value_R.ub[5]]

    ls = false

    filtered_results_scaled = filter(real_results_scaled) do res
        all(lb_scaled .<= res .<= ub_scaled) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
    end

    if isempty(filtered_results_scaled)
        @error "No physical solutions found by HC. We project all real results to bounded results instead"
        ls = true
        filtered_results_scaled = Vector{Float64}[]
        for result in real_results_scaled
            bound_result = Logic_R.project_to_bounds(result, lb_scaled, ub_scaled, B[1])
            push!(filtered_results_scaled, bound_result)
        end
    end

    try_filter = filter(filtered_results_scaled) do res
        (res[4] > res[5]) && (res[1] != 0) && (res[2] != 0) && (res[3] != 0)
    end

    if !isempty(try_filter)
        filtered_results_scaled = try_filter
    end

    best_result_scaled, best_err = Logic_R.best_solution(filtered_results_scaled, I_data, B..., t_scaled)
    best_result = to_physical(best_result_scaled, T)
    err = Logic_R.get_error(best_result)
    err_I_data = Logic_R.RSS_I_data(I_data, I)

    I_fit_test = Logic_R.residual(best_result_scaled, B..., t_scaled)
    rss_test = sum((I_fit_test .- I_data).^2)
    @assert best_err ≈ rss_test "RSS Mismatch during HC"

    println("  Method: $method")
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
    println("  RSS (sum((I_data .- I).^2)): $err_I_data")

    if ls
        println(" ====== Performing LS ======")
        u0 = best_result_scaled

        function model(x, p)
            return Logic_R.residual(p, B..., x)
        end

        fit = curve_fit(model, t_scaled, I_data, u0, lower=lb_scaled, upper=ub_scaled)
        estimated = to_physical(fit.param, T)

        err_list = Logic_R.get_error(estimated)
        cost = sum((Value_R.true_vals .- estimated).^2)
        I_fit = Logic_R.residual(fit.param, B..., t_scaled)
        rss = sum((I_fit .- I_data).^2)

        println("  Variables after LS: $estimated")
        println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err_list")
        println("  Cost: $cost")
        println("  RSS (sum((I_hat .- I_data).^2)): $rss")
    end
end


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

    filtered_results_scaled = Vector{Float64}[]
    for result in real_results_scaled
        bound_result = Logic_R.project_to_bounds(result, lb_scaled, ub_scaled, B[1])
        push!(filtered_results_scaled, bound_result)
    end

    #=
    try_filter = filter(filtered_results_scaled) do res
        (res[4] > res[5]) && (res[1] != 0) && (res[2] != 0) && (res[3] != 0)
    end

    if !isempty(try_filter)
        filtered_results_scaled = try_filter
    end
    =#

    best_result_scaled, best_err = Logic_R.best_solution(filtered_results_scaled, I_data, B..., t_scaled)
    best_result = to_physical(best_result_scaled, T)
    err = Logic_R.get_error(best_result)
    err_I_data = Logic_R.RSS_I_data(I_data, I)

    println("  Method: $method")
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
    println("  RSS (sum((I_data .- I).^2)): $err_I_data")

    println()
    println(" ====== Performing LS ======")
    u0 = best_result_scaled

    function model(x, p)
        return Logic_R.residual(p, B..., x)
    end

    fit = curve_fit(model, t_scaled, I_data, u0, lower=lb_scaled, upper=ub_scaled)
    estimated = to_physical(fit.param, T)

    err_list = Logic_R.get_error(estimated)
    cost = sum((Value_R.true_vals .- estimated).^2)
    I_fit = Logic_R.residual(fit.param, B..., t_scaled)
    rss = sum((I_fit .- I_data).^2)

    println("  Variables after LS: $estimated")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err_list")
    println("  Cost: $cost")
    println("  RSS (sum((I_hat .- I_data).^2)): $rss")
end


function comp_best_result_complex(t_scaled::Vector, T::Float64, I::Vector, I_data::Vector, vars::Vector, method::String; complex_tolerance::Float64 = 1e-10)
    B = Logic_R.get_blocks(I_data, t_scaled, method)
    I_hat = Logic_R.residual(vars, B..., t_scaled)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results_scaled = Vector{Float64}[]

    for sol in solutions(result)
        if maximum(abs.(imag.(sol))) < complex_tolerance
            push!(real_results_scaled, real.(sol))
        end
    end

    lb_scaled = [Value_R.lb[1]*T, Value_R.lb[2]*T, Value_R.lb[3]*T, Value_R.lb[4], Value_R.lb[5]]
    ub_scaled = [Inf, Inf, Inf, Value_R.ub[4], Value_R.ub[5]]

    filtered_results_scaled = Vector{Float64}[]

    for result in real_results_scaled
        bound_result = Logic_R.project_to_bounds(result, lb_scaled, ub_scaled, B[1])
        push!(filtered_results_scaled, bound_result)
    end

    best_result_scaled, best_err = Logic_R.best_solution(filtered_results_scaled, I_data, B..., t_scaled)
    best_result = to_physical(best_result_scaled, T)
    err = Logic_R.get_error(best_result)
    err_I_data = Logic_R.RSS_I_data(I_data, I)

    I_fit_test = Logic_R.residual(best_result_scaled, B..., t_scaled)
    rss_test = sum((I_fit_test .- I_data).^2)
    @assert best_err ≈ rss_test "RSS Mismatch during HC"

    println("  Method: $method")
    println("  Variables: $best_result")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
    println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
    println("  RSS (sum((I_data .- I).^2)): $err_I_data")

    return best_err

    #=
    println()
    println(" ====== Performing LS ======")
    u0 = best_result_scaled

    function model(x, p)
        return Logic_R.residual(p, B..., x)
    end

    fit = curve_fit(model, t_scaled, I_data, u0, lower=lb_scaled, upper=ub_scaled)
    estimated = to_physical(fit.param, T)

    err_list = Logic_R.get_error(estimated)
    cost = sum((Value_R.true_vals .- estimated).^2)
    I_fit = Logic_R.residual(fit.param, B..., t_scaled)
    rss = sum((I_fit .- I_data).^2)

    println("  Variables after LS: $estimated")
    println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err_list")
    println("  Cost: $cost")
    println("  RSS (sum((I_hat .- I_data).^2)): $rss")
    =#
end


function comp_best_result_ls(t_scaled::Vector, T::Float64, I::Vector, I_data::Vector, vars::Vector, method::String; complex_tolerance::Float64 = 1e-5, if_print=true)
    B = Logic_R.get_blocks(I_data, t_scaled, method)

    function model(x, p)
        return Logic_R.residual(p, B..., x)
    end

    I_hat = Logic_R.residual(vars, B..., t_scaled)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results_scaled = Vector{Float64}[]
    complex_results_scaled = Vector{Complex{Float64}}[]

    #=
    for sol in solutions(result)
        if maximum(abs.(imag.(sol))) < complex_tolerance
            push!(real_results_scaled, real.(sol))
        else
            push!(complex_results_scaled, sol)
        end
    end
    =#

    for sol in solutions(result)
        push!(real_results_scaled, real.(sol))
    end

    lb_scaled = [Value_R.lb[1]*T, Value_R.lb[2]*T, Value_R.lb[3]*T, Value_R.lb[4], Value_R.lb[5]]
    ub_scaled = [Inf, Inf, Inf, Value_R.ub[4], Value_R.ub[5]]

    filtered_results_scaled = Vector{Float64}[]
    after_projection_results_scaled = Vector{Float64}[]

    for result in real_results_scaled
        bound_result = Logic_R.project_to_bounds(result, lb_scaled, ub_scaled, B[1])
        fit = curve_fit(model, t_scaled, I_data, bound_result, lower=lb_scaled, upper=ub_scaled)
        push!(filtered_results_scaled, fit.param)
        push!(after_projection_results_scaled, bound_result)
    end

    best_result_scaled, best_err = Logic_R.best_solution(filtered_results_scaled, I_data, B..., t_scaled)
    best_result = to_physical(best_result_scaled, T)
    err = Logic_R.get_error(best_result)
    err_I_data = Logic_R.RSS_I_data(I_data, I)

    I_fit_test = Logic_R.residual(best_result_scaled, B..., t_scaled)
    rss_test = sum((I_fit_test .- I_data).^2)
    @assert best_err ≈ rss_test "RSS Mismatch during HC"

    if if_print
        # print all solutions after projection and LS if our RSS is too far from baseline
        println("Num of (approx, with tolerance $complex_tolerance) real solution: ", length(real_results_scaled))
        if best_err > 4
            for (solution_scaled, solution_projection_scaled, solution_projection_ls_scaled) in zip(real_results_scaled, after_projection_results_scaled, filtered_results_scaled)
                solution = to_physical(solution_scaled, T)
                solution_projection = to_physical(solution_projection_scaled, T)
                solution_projection_ls =  to_physical(solution_projection_ls_scaled, T)
                println("Original: $solution \n After projection: $solution_projection \n After projection and LS: $solution_projection_ls")
                println("-"^20)
            end
            println("="^20)
            println("Rest complex root")
            for complex_sol_scaled in complex_results_scaled
                complex_sol = to_physical(complex_sol_scaled, T)
                println(complex_sol)
            end
        end

        println("  Method: $method")
        println("  Variables: $best_result")
        println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $err")
        println("  RSS (sum((I_hat .- I_data).^2)): $best_err")
        println("  RSS (sum((I_data .- I).^2)): $err_I_data")
    end

    return (
        method = method,
        best_result = best_result,
        parameter_err = err,
        RSS_Ihat_Idata = best_err,
        RSS_Idata_I = err_I_data
    )
end


function check_degenerate(t_scaled::Vector{Float64}, T::Float64, I::Vector{Float64}, noise::Float64, vars::Vector, method::String, num_of_iter::Int; RSS_tolerance::Float64=4.0)
    num_of_degenerate_result = 0
    total_RSS_Ihat_Idata = 0
    total_valid_RSS_Ihat_Idata = 0
    total_RSS_Idata_I = 0
    param_alpha_list = Float64[]
    param_sigma_list = Float64[]
    param_gamma_list = Float64[]
    param_S0_list = Float64[]
    param_E0_list = Float64[]

    for _ in 1:num_of_iter
        I_data = I .+ noise .* I .* randn(length(I))
        result_tuple = comp_best_result_ls(t_scaled, T, I, I_data, vars, method; if_print=false)

        best_result = result_tuple.best_result
        push!(param_alpha_list, best_result[1])
        push!(param_sigma_list, best_result[2])
        push!(param_gamma_list, best_result[3])
        push!(param_S0_list, best_result[4])
        push!(param_E0_list, best_result[5])

        RSS_Ihat_Idata = result_tuple.RSS_Ihat_Idata
        RSS_Idata_I = result_tuple.RSS_Idata_I
        parameter_err = result_tuple.parameter_err

        total_RSS_Ihat_Idata += RSS_Ihat_Idata
        total_RSS_Idata_I += RSS_Idata_I

        if RSS_Ihat_Idata > RSS_tolerance
            println("  INVALID RESULT")
            println("  Method: $method")
            println("  Variables: $best_result")
            println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $parameter_err")
            println("  RSS (sum((I_hat .- I_data).^2)): $RSS_Ihat_Idata")
            println("  RSS (sum((I_data .- I).^2)): $RSS_Idata_I")
            num_of_degenerate_result += 1
        else
            total_valid_RSS_Ihat_Idata += RSS_Ihat_Idata
        end
    end

    average_RSS_Ihat_Idata = total_RSS_Ihat_Idata / num_of_iter
    num_valid = num_of_iter - num_of_degenerate_result
    average_valid_RSS_Ihat_Idata = num_valid > 0 ? total_valid_RSS_Ihat_Idata / num_valid : NaN
    average_RSS_Idata_I = total_RSS_Idata_I / num_of_iter
    rate = num_of_degenerate_result / num_of_iter

    alpha_mean, alpha_std = mean(param_alpha_list), std(param_alpha_list)
    sigma_mean, sigma_std = mean(param_sigma_list), std(param_sigma_list)
    gamma_mean, gamma_std = mean(param_gamma_list), std(param_gamma_list)
    S0_mean,    S0_std    = mean(param_S0_list),    std(param_S0_list)
    E0_mean,    E0_std    = mean(param_E0_list),    std(param_E0_list)

    return (
        noise = noise,
        method = method,
        num_of_iter = num_of_iter,
        rate = rate,
        average_RSS_Ihat_Idata = average_RSS_Ihat_Idata,
        average_valid_RSS_Ihat_Idata = average_valid_RSS_Ihat_Idata,
        average_RSS_Idata_I = average_RSS_Idata_I,
        param_alpha_list = param_alpha_list,
        param_sigma_list = param_sigma_list,
        param_gamma_list = param_gamma_list,
        param_S0_list = param_S0_list,
        param_E0_list = param_E0_list,

        alpha_mean = alpha_mean, alpha_std = alpha_std,
        sigma_mean = sigma_mean, sigma_std = sigma_std,
        gamma_mean = gamma_mean, gamma_std = gamma_std,
        S0_mean = S0_mean, S0_std = S0_std,
        E0_mean = E0_mean, E0_std = E0_std
    )
end


function _print_degenerate_summary(result)
    noise = result.noise
    println("========== Degenerate Check Summary (noise level: $noise) ==========")
    println("Method: ", result.method)
    println("Number of iterations: ", result.num_of_iter)
    println("Degenerate rate: ", round(result.rate * 100, digits=2), "%")

    println()
    println("Average RSS (all runs): ",
        round(result.average_RSS_Ihat_Idata, sigdigits=6))

    println("Average RSS (valid runs only): ",
        isnan(result.average_valid_RSS_Ihat_Idata) ?
        "NaN (no valid runs)" :
        round(result.average_valid_RSS_Ihat_Idata, sigdigits=6))

    println("Average RSS (I_data vs I): ",
        round(result.average_RSS_Idata_I, sigdigits=6))

    println("========== Estimated Parameter (mean ± std) ==========")
    println("α   : ", round(result.alpha_mean, sigdigits=6), " ± ", round(result.alpha_std, sigdigits=6))
    println("σ   : ", round(result.sigma_mean, sigdigits=6), " ± ", round(result.sigma_std, sigdigits=6))
    println("γ   : ", round(result.gamma_mean, sigdigits=6), " ± ", round(result.gamma_std, sigdigits=6))
    println("S0  : ", round(result.S0_mean, sigdigits=6),    " ± ", round(result.S0_std, sigdigits=6))
    println("E0  : ", round(result.E0_mean, sigdigits=6),    " ± ", round(result.E0_std, sigdigits=6))

    println("==============================================")
end


function print_degenerate_summary(t_scaled::Vector{Float64}, T::Float64, I::Vector{Float64}, noise::Float64, vars::Vector, num_of_iter::Int)
    result_T = check_degenerate(t_scaled, T, I, noise, vars, "T", num_of_iter)
    result_S = check_degenerate(t_scaled, T, I, noise, vars, "S", num_of_iter)

    _print_degenerate_summary(result_T)
    _print_degenerate_summary(result_S)
end


function HC_LS(t_scaled::Vector, T::Float64, I::Vector, I_data::Vector, vars::Vector, method::String; if_print=false)
    B = Logic_R.get_blocks(I_data, t_scaled, method)

    function model(x, p)
        return Logic_R.residual(p, B..., x)
    end

    I_hat = Logic_R.residual(vars, B..., t_scaled)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results_scaled = Vector{Float64}[]

    for sol in solutions(result)
        push!(real_results_scaled, real.(sol))
    end

    lb_scaled = [Value_R.lb[1]*T, Value_R.lb[2]*T, Value_R.lb[3]*T, Value_R.lb[4], Value_R.lb[5]]
    ub_scaled = [Inf, Inf, Inf, Value_R.ub[4], Value_R.ub[5]]

    filtered_results_scaled = Vector{Float64}[]
    after_projection_results_scaled = Vector{Float64}[]

    for r in real_results_scaled
        bound_result = Logic_R.project_to_bounds(r, lb_scaled, ub_scaled, B[1])
        fit = curve_fit(model, t_scaled, I_data, bound_result, lower=lb_scaled, upper=ub_scaled)
        push!(filtered_results_scaled, fit.param)
        push!(after_projection_results_scaled, bound_result)
    end

    best_result_scaled, RSS_Ihat_Idata = Logic_R.best_solution(filtered_results_scaled, I_data, B..., t_scaled)
    best_result = to_physical(best_result_scaled, T)
    parameter_err = Logic_R.get_error(best_result)
    RSS_Idata_I = Logic_R.RSS_I_data(I_data, I)

    if if_print
        println("  Method: $method")
        println("  Variables: $best_result")
        println("  Parameter err (abs.(est .- true_value) ./ true_value .* 100) $parameter_err")
        println("  RSS (sum((I_hat .- I_data).^2)): $RSS_Ihat_Idata")
        println("  RSS (sum((I_data .- I).^2)): $RSS_Idata_I")
    end

    return (
        method = method,
        best_result = best_result,
        parameter_err = parameter_err,
        RSS_Ihat_Idata = RSS_Ihat_Idata,
        RSS_Idata_I = RSS_Idata_I
    )
end


function HC_LS_parameter_analysis(t_scaled::Vector, T::Float64, I::Vector, noise::Float64, vars::Vector, num_of_iter::Int)
    result_dict = Dict{String, Dict{String, Vector{Float64}}}()
    method_list = String["T", "S"]
    parameter_list = String["α", "σ", "γ", "S0", "E0"]
    metrics = String["α_err", "σ_err", "γ_err", "S0_err", "E0_err", "RSS_Ihat_Idata", "RSS_Idata_I"]
    metrics_one = String["RSS_Ihat_Idata", "RSS_Idata_I"]
    metrics_two = String["α_err", "σ_err", "γ_err", "S0_err", "E0_err"]

    for method in method_list
        result_dict[method] = Dict{String, Vector{Float64}}()
        for param in parameter_list
            result_dict[method][param] = Float64[]
        end

        for m in metrics
            result_dict[method][m] = Float64[]
        end
    end

    for _ in 1:num_of_iter

        I_data = I .+ noise .* I .* randn(length(I))

        for method in method_list
            result_method = HC_LS(t_scaled, T, I, I_data, vars, method)

            for (i, param) in enumerate(parameter_list)
                push!(result_dict[method][param], result_method.best_result[i])
            end

            for m in metrics_one
                push!(result_dict[method][m], getfield(result_method, Symbol(m)))
            end

            for (i, m) in enumerate(metrics_two)
                push!(result_dict[method][m], result_method.parameter_err[i])
            end
        end
    end

    println("========== HC + LS Parameter Analysis (noise = $noise) ==========")
    println("Number of iterations: $num_of_iter")
    println()

    for method in method_list
        println("---------- Method: $method ----------")

        println("Average RSS (I_hat vs I_data): ",
            round(mean(result_dict[method]["RSS_Ihat_Idata"]), sigdigits=6),
            " ± ",
            round(std(result_dict[method]["RSS_Ihat_Idata"]), sigdigits=6))

        println("Average RSS (I_data vs I): ",
            round(mean(result_dict[method]["RSS_Idata_I"]), sigdigits=6),
            " ± ",
            round(std(result_dict[method]["RSS_Idata_I"]), sigdigits=6))

        println()
        println("Estimated Parameters (mean ± std)")

        for param in parameter_list
            μ = mean(result_dict[method][param])
            σ = std(result_dict[method][param])
            println(param, " : ",
                round(μ, sigdigits=6),
                " ± ",
                round(σ, sigdigits=6))
        end

        println()
        println("Parameter Errors (%) (mean ± std)")

        for err in metrics_two
            μ = mean(result_dict[method][err])
            σ = std(result_dict[method][err])
            println(err, " : ",
                round(μ, sigdigits=6),
                " ± ",
                round(σ, sigdigits=6))
        end

        println("==============================================")
        println()
    end
end

## CHECK THIS WITH PROF
# Since before in sanity check integration, the integrals we are checking is not really the blocks, but the integrals appeared in the blocks
# So if we rescale based on the sanity check integration, our blocks may still have scales with large difference.
# Since the blocks B1 - B6 also have scale rules, we could apply auto selection on T based on their maximum value.
# So we can chance from choosing T based on those integrals we analyze, to choosing T based on the actual blocks.

function select_T(I, t; method="S", m_min=-6, m_max=6)
    I0, B1, B2, B3, B4, B5, B6 = Logic_R.get_blocks(I, t, method)

    s = [
        B2[end],  # scales as 1/T
        B3[end],  # scales as 1/T
        B4[end],  # scales as 1/T^2
        B5[end],  # scales as 1/T^2
        B6[end],  # scales as 1/T^3
    ]
    powers = [1, 1, 2, 2, 3]   # exponent of T in denominator

    best_m = nothing
    best_score = Inf

    for m in m_min:m_max
        T = 10.0^m  # use float to allow negative m cleanly
        scaled = [ s[j] / (T^powers[j]) for j in eachindex(s) ]
        logs = log10.(scaled .+ eps())  # eps() avoids log(0)
        score = var(logs)

        if score < best_score
            best_score = score
            best_m = m
        end
    end

    best_T = 10.0^best_m
    final_scaled = [ s[j] / (best_T^powers[j]) for j in eachindex(s) ]

    return best_T, final_scaled
end


function main()
    t = collect(0.0:10.0:1000.0)
    S_, E_, I_, R_ = Logic_R.simulate_seir(t)
    T, _ = select_T(I_, t)
    t_scaled = t ./ T
    S, E, I, R = Logic_R.simulate_seir_HC_LS(t_scaled, T, plot=true)
    noise = 0.01
    I_data = I .+ noise .* I .* randn(length(I))

    # comp_best_result_ls(t_scaled, T, I, I_data, variables, "S")
    # comp_best_result_ls(t_scaled, T, I, I_data, variables, "T")

    # result = check_degenerate(t_scaled, T, I, noise, variables, "S", 200)
    # _print_degenerate_summary(result)

    # print_degenerate_summary(t_scaled, T, I, noise, variables, 200)

    # println("Begin??: ")
    # begin_or_not = readline()
    #=
        println("True parameters: ", Value_R.true_vals)
        noise_level_list = [0,0, 0.001, 0.01, 0.05]
        for noise in [0.05]
            HC_LS_parameter_analysis(t_scaled, T, I, noise, variables, 10)
        end
    =#


    #=
    noise_steps = 41
    noise_levels = [0.005 * i for i in 0:noise_steps-1]      # 0 to 0.2
    x_noise_percent = [0.5 * i for i in 0:noise_steps-1]     # 0% to 20%
    Logic_R.noise_level_analysis_HC_LS(I, t_scaled, T, variables, noise_levels, x_noise_percent)
    =#

    #=
    num_of_datapoints = [i for i in 10:5:100]
    Logic_R.num_of_datapoints_analysis_HC_LS(num_of_datapoints, 0.01, T, variables)
    =#


    k_points = [10 * i for i in 1:1:10]
    Logic_R.run_experiments_k_points_HC_LS(k_points, noise, I, t_scaled, T, variables)



    println()
end

main()
