# homotopy_continuation.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 03/02/2026
=#

using Revise
using HomotopyContinuation, DynamicPolynomials, JLD2
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
global vars = [α, σ, γ, S0, E0]

function comp_I_hat(I_data, t)
    B1, B2, B3, B4, B5, B6 = Logic.get_blocks(I_data, t)
    I0 = I_data[1]
    C1 = σ * (E0 - I0) .* t .* B1
    C2 = - (γ + σ) .* B2
    C3 = - 0.5 * α .* B3
    C4 = (α * σ * (S0 + E0 + I0) - σ * γ) .* B4
    C5 = - α * (γ + σ) .* B5
    C6 = - 0.5 * α * σ * γ .* B6
    I_hat = C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6

    return I_hat
end

function comp_results(I_hat::Vector, I_data::Vector, vars::Vector; save_name::String="real_solution_homotopy.jld2", save::Bool=true)
    J = sum((I_hat .- I_data).^2)
    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)
    result = HomotopyContinuation.solve(C)
    real_results = real_solutions(result)
    if save
        path = joinpath(@__DIR__, save_name)
        jldsave(path; real_results)
    end
    return real_results
end

function noise_level_comparison(I::Vector, noise_levels::Vector, vars::Vector, t; save_name::String="noise_results.jld2")
    path = joinpath(@__DIR__, save_name)

    if isfile(path)
        rm(path)
    end

    for noise in noise_levels
        I_data = I .+ noise .* I .* randn(length(I))
        I_hat = comp_I_hat(I_data, t)
        real_results = comp_results(I_hat, I_data, vars, save=false)
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
            err_list = Logic.get_error(estimated_vals, true_vals)


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

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t, plot=false)
    noise_levels = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    #noise_level_comparison(I, noise_levels, vars, t)
    print_param_values_vs_noise(noise_levels)
    plot_param_values_vs_noise(noise_levels)
end

main()