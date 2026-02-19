# integration_timerescale_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/02/2026
=#

#=
UNFIXED BUG IN THIS FILE, RESULT NOT ACCURATE
=#

using Printf, DifferentialEquations, Plots, NumericalIntegration
include("SEIRModels_ratio.jl")
using .Logic_R
using .Value_R


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


function odes_scaled!(du, u, p, τ)
    S, E, I, R, I_int, I_int_sq, I_int_int_sq, I_int_B4, I_int_B5 = u
    α, σ, γ = p
    T = Value_R.T   # or pass T as parameter if you prefer

    # SEIR dynamics (multiply whole RHS by T)
    du[1] = T * (- α * S * I)
    du[2] = T * (α * S * I - σ * E)
    du[3] = T * (σ * E - γ * I)
    du[4] = T * (γ * I)

    # Integrals (also multiplied by T)
    du[5] = T * I
    du[6] = T * I^2
    du[7] = T * I_int^2

    # Important: t = T * τ
    du[8] = T * ((T * τ) * I)
    du[9] = T * ((T * τ) * I^2)
end


function validation_check(I_data, t)::Dict{String, Dict{String, Vector{Float64}}}
    results = Dict{String, Dict{String, Vector{Float64}}}()

    results["Trapezoidal"] = Dict{String, Vector{Float64}}()
    results["Simpson"] = Dict{String, Vector{Float64}}()
    results["Baseline"] = Dict{String, Vector{Float64}}()

    # 1. cumul_integrate by Numerical Integration (Trapezoidal)
    I_int_1 = cumul_integrate(t, I_data)
    I_int_sq_1 = cumul_integrate(t, I_data.^2)
    I_int_int_sq_1 = cumul_integrate(t, (I_int_1).^2)
    I_int_B4_1 = cumul_integrate(t, t .* I_data)
    I_int_B5_1 = cumul_integrate(t, t .* (I_data.^2))

    results["Trapezoidal"]["I_int"] = I_int_1
    results["Trapezoidal"]["I_int_sq"] = I_int_sq_1
    results["Trapezoidal"]["I_int_int_sq"] = I_int_int_sq_1
    results["Trapezoidal"]["I_int_B4"] = I_int_B4_1
    results["Trapezoidal"]["I_int_B5"] = I_int_B5_1

    # 2. Simpson
    I_int_2 = Logic_R.cumintegrate(t, I_data)
    I_int_sq_2 = Logic_R.cumintegrate(t, I_data.^2)
    I_int_int_sq_2 = Logic_R.cumintegrate(t, (I_int_2).^2)
    I_int_B4_2 = Logic_R.cumintegrate(t, t .* I_data)
    I_int_B5_2 = Logic_R.cumintegrate(t, t .* (I_data.^2))

    results["Simpson"]["I_int"] = I_int_2
    results["Simpson"]["I_int_sq"] = I_int_sq_2
    results["Simpson"]["I_int_int_sq"] = I_int_int_sq_2
    results["Simpson"]["I_int_B4"] = I_int_B4_2
    results["Simpson"]["I_int_B5"] = I_int_B5_2

    # 3. Numerical Integration
    prob = ODEProblem(odes_scaled!, [Value_R.S0, Value_R.E0, Value_R.I0, Value_R.R0, 0.0, 0.0, 0.0, 0.0, 0.0], (t[1], t[end]), Value_R.p_true)
    sol = DifferentialEquations.solve(prob, saveat=t, reltol=1e-15, abstol=1e-15)
    sol_arr = Array(sol)
    I_int_3 = sol_arr[5, :]
    I_int_sq_3 = sol_arr[6, :]
    I_int_int_sq_3 = sol_arr[7, :]
    I_int_B4_3 = sol_arr[8, :]
    I_int_B5_3 = sol_arr[9, :]

    results["Baseline"]["I_int"] = I_int_3
    results["Baseline"]["I_int_sq"] = I_int_sq_3
    results["Baseline"]["I_int_int_sq"] = I_int_int_sq_3
    results["Baseline"]["I_int_B4"] = I_int_B4_3
    results["Baseline"]["I_int_B5"] = I_int_B5_3

    return results
end


function validation_print(I, noise_level, t)
    I_data = I .+ noise_level .* I .* randn(length(I))
    results = validation_check(I_data, t)

    println("\n" * "="^95)
    @printf(" VALIDATION CHECK | Noise: %0.3f%% | N: %d points\n", noise_level * 100, length(t))
    println("="^95)
    @printf("%-15s | %-12s | %-12s | %-12s | %-12s | %-8s\n",
            "Metric", "Max Err (T)", "Max Err (S)", "Sum Err (T)", "Sum Err (S)", "Gain (Sum Err(T) / Sum Err(S))")
    println("-"^95)

    keys_to_check = String["I_int", "I_int_sq", "I_int_int_sq", "I_int_B4", "I_int_B5"]

    for key in keys_to_check
        truth = results["Baseline"][key]
        trap  = results["Trapezoidal"][key]
        simp  = results["Simpson"][key]

        res_trap = abs.(trap .- truth)
        res_simp = abs.(simp .- truth)

        max_t = maximum(res_trap)
        max_s = maximum(res_simp)

        sum_t = sum(res_trap)
        sum_s = sum(res_simp)

        # Accuracy Gain (Ratio of Total Trapezoidal error to Simpson error)
        gain = sum_t / sum_s

        @printf("%-15s | %-12.4e | %-12.4e | %-12.4e | %-12.4e | %-8.1fx\n",
                key, max_t, max_s, sum_t, sum_s, gain)
    end
    println("="^95)
end


function validation_plot_complete(I, noise_level, t; plot_I_data=false)
    I_data = I .+ noise_level .* I .* randn(length(I))
    results = validation_check(I_data, t)

    keys_to_check = String["I_int", "I_int_sq", "I_int_int_sq", "I_int_B4", "I_int_B5"]
    plot_list = Plots.Plot[]

    for key in keys_to_check
        truth = results["Baseline"][key]
        trap  = results["Trapezoidal"][key]
        simp  = results["Simpson"][key]

        # Subplot The Values
        p_val = plot(t, truth, label="Baseline", title="Value: $key", lw=2, color=:black)
        plot!(p_val, t, trap, label="Trapezoidal", ls=:dash, color=:red)
        plot!(p_val, t, simp, label="Simpson", ls=:dot, color=:blue)

        # Subplot The Error
        p_err = plot(t, trap .- truth, label="Trap Error", title="Error: $key", color=:red)
        plot!(p_err, t, simp .- truth, label="Simp Error", color=:blue)
        hline!(p_err, [0], color=:black, alpha=0.3, label="")

        # Subplot The Error
        p_err_I_data = plot(t, trap .- truth, label="Trap Error", title="Error: $key", color=:red)
        plot!(p_err_I_data, t, simp .- truth, label="Simp Error", color=:blue)
        hline!(p_err_I_data, [0], color=:black, alpha=0.3, label="")
        if plot_I_data
            plot!(p_err_I_data, t, I, label="I_data", color=:grey)
        end

        # Subplot The Error Percentage
        valid_idx = abs.(truth) .> 1e-6
        p_err_percentage = plot(t[valid_idx], abs.((trap[valid_idx] .- truth[valid_idx]) ./ truth[valid_idx]),
            label="Trap %",
            color=:red,
            yscale=:log10)

        plot!(p_err_percentage, t[valid_idx], abs.((simp[valid_idx] .- truth[valid_idx]) ./ truth[valid_idx]),
            label="Simp %",
            color=:blue)

        push!(plot_list, p_val, p_err, p_err_I_data, p_err_percentage)
    end

    final_plot = plot(plot_list..., layout=(length(keys_to_check), 4),
                      size=(1200, 300 * length(keys_to_check)),
                      plot_title="Noise Level: $(noise_level*100)%")

    display(final_plot)
    return final_plot
end


function main()
    t = collect(0.0:10.0:1000.0)
    T = 100.0
    t_scaled = t ./ T
    S, E, I, R = simulate_seir(t_scaled, T)
    noise_levels = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    noise_level = [0.001]
    for noise in noise_level
        final_plot = validation_plot_complete(I, noise, t_scaled, plot_I_data=true)
        savefig(final_plot, "sanity_check_complete.pdf")
    end

    for noise in noise_levels
        validation_print(I, noise, t_scaled)
    end
end


main()
