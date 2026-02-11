# homotopy_continuation.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 03/02/2026
=#

using Revise
using HomotopyContinuation, DynamicPolynomials, JLD2, Random
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const variables = [α, σ, γ, S0, E0]

function comp_results(t::Vector, I_data::Vector, vars::Vector, method::String)
    B = Logic.get_blocks(I_data, t, method)
    I_hat = Logic.residual(vars, B..., t)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C)
    real_results = real_solutions(result)

    filtered_results = filter(real_results) do res
        all(Value.lb .<= res .<= Value.ub) && (res[2] > res[3])
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC, returning un-filtered real results"
        return real_results
    end

    return filtered_results
end

function noise_level_comparison(I::Vector, noise_levels::Vector, vars::Vector, t; save_name::String="noise_results.jld2")
    path = joinpath(@__DIR__, save_name)

    if isfile(path)
        rm(path)
    end

    for noise in noise_levels
        I_data = I .+ noise .* I .* randn(length(I))
        real_results = comp_results(t, I_data, vars, save=false)
        best_result, best_cost = Logic.best_solution(real_results, I_data, t)
        jldopen(path, "a+") do file
            key_name = "noise_$noise"
            file[key_name] = best_result, best_cost
        end
    end
end

function print_param_values_vs_noise(noise_levels::Vector, save_name::String="noise_results.jld2")
    path = joinpath(@__DIR__, save_name)
    data = load(path)

    labels = String["α", "σ", "γ", "S0", "E0"]
    true_vals = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0]

    for noise in sort(noise_levels)
        key = "noise_$(noise)"
        if haskey(data, key)
            estimated_vals = data[key][1]
            cost = data[key][2]
            err_list = Logic.get_error(estimated_vals)


            println("======================== Noise Level: $noise ========================")
            for (label, t_val, e_val, err_val) in zip(labels, true_vals, estimated_vals, err_list)
                println("----- $label -----")
                println("True: $t_val | Result: $e_val | Error: $err_val")
            end

            println("Total Residual Sum of Squares (RSS): $cost")
        end
    end
end

function plot_param_values_vs_noise(noise_levels::Vector, save_name::String="noise_results.jld2")
    path = joinpath(@__DIR__, save_name)
    data = load(path)

    labels = String["α", "σ", "γ", "S0", "E0"]
    true_vals = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0]

    # Prepare storage for plotting
    plot_list = []
    valid_noises = sort(noise_levels)

    # Create one subplot for each of the 5 parameters
    for i in 1:5
        estimates = Float64[]
        for noise in valid_noises
            key = "noise_$(noise)"
            if haskey(data, key)
                push!(estimates, data[key][1][i])
            end
        end

        # Create the subplot for parameter i
        p = plot(valid_noises, estimates,
                 title=labels[i],
                 ylabel="Value",
                 xlabel="Noise",
                 marker=:o,
                 legend=false,
                 color=:blue)

        # Add the horizontal line for the ground truth
        hline!(p, [true_vals[i]], color=:red, lw=2, linestyle=:dash)

        push!(plot_list, p)
    end

    final_plt = plot(plot_list..., layout=(2, 3), size=(1200, 800))
    display(final_plt)
end

function comp_best_result(t::Vector, I::Vector, I_data::Vector, vars::Vector, method::String)
    B = Logic.get_blocks(I_data, t, method)
    I_hat = Logic.residual(vars, B..., t)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C)
    real_results = real_solutions(result)

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
    comp_best_result(t, I, I_data, variables, "T")
end

main()