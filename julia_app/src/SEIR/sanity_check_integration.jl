# sanity_check_integration.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 05/02/2026
=#

include("SEIRModels.jl")
using .Logic
using .Value
using Printf, DifferentialEquations

# 2. Simpson()
function cumintegrate(x, y, method=SimpsonEven())
    n = length(x)
    output = zeros(promote_type(eltype(x), eltype(y)), n)

    if n == 1
        error("cumintegrate requires at least 2 points")
    end

    if n == 2
        output[2] = (x[2] - x[1]) * (y[1] + y[2]) / 2
        return output
    end

    for i in 3:2:n
        x1 = x[i-2]
        x2 = x[i-1]
        x3 = x[i]
        y1 = y[i-2]
        y2 = y[i-1]
        y3 = y[i]

        h1 = x2 - x1
        h2 = x3 - x2
        h_total = x3 - x1

        # use the standard Simpson's 1/3 rule
        # from http://www.msme.us/2017-2-1.pdf formula 6
        output[i] = output[i-2] + (h_total / 6) * (
            (2 - h2 / h1) * y1 +
            (h_total^2 / (h1 * h2)) * y2 +
            (2 - h1 / h2) * y3
        )

        # to compute output[i-1] we use the formula for scipy.integrate.cumulative_simpson
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html#rb3a817c91225-2
        # from http://www.msme.us/2017-2-1.pdf formula 8
        output[i-1] = output[i-2] + (h1 / 6) * (
            (3 - h1 / h_total) * y1 +
            (3 + h1^2 / (h2 * h_total) + h1 / h_total) * y2 -
            (h1^2 / (h2 * h_total)) * y3
        )
    end

    if iseven(n) && n >= 3
        # Use the last 3 points
        x1, x2, x3 = x[n-2], x[n-1], x[n]
        y1, y2, y3 = y[n-2], y[n-1], y[n]
        h1, h2 = x2 - x1, x3 - x2
        h_total = x3 - x1

        # use formula 8 to compute the last point integration
        # notice we need to do for these three points: total_simpson - first_half
        total_simpson = (h_total / 6) * (
            (2 - h2 / h1) * y1 +
            (h_total^2 / (h1 * h2)) * y2 +
            (2 - h1 / h2) * y3
        )

        first_half = (h1 / 6) * (
            (3 - h1 / h_total) * y1 +
            (3 + h1^2 / (h2 * h_total) + h1 / h_total) * y2 -
            (h1^2 / (h2 * h_total)) * y3
        )

        output[n] = output[n-1] + (total_simpson - first_half)
    end

    return output
end

# 3. Numerical Integration (accurate baseline, no noise)
function odes!(du, u, p, t)
    S, E, I, R, I_int, I_int_sq, I_int_int_sq, I_int_B4, I_int_B5 = u
    α, σ, γ = p
    du[1] = - α * S * I
    du[2] = α * S * I - σ * E
    du[3] = σ * E - γ * I
    du[4] = γ * I

    du[5] = I
    du[6] = I^2
    du[7] = I_int^2
    du[8] = t * I
    du[9] = t * (I^2)
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
    I_int_2 = cumintegrate(t, I_data)
    I_int_sq_2 = cumintegrate(t, I_data.^2)
    I_int_int_sq_2 = cumintegrate(t, (I_int_2).^2)
    I_int_B4_2 = cumintegrate(t, t .* I_data)
    I_int_B5_2 = cumintegrate(t, t .* (I_data.^2))

    results["Simpson"]["I_int"] = I_int_2
    results["Simpson"]["I_int_sq"] = I_int_sq_2
    results["Simpson"]["I_int_int_sq"] = I_int_int_sq_2
    results["Simpson"]["I_int_B4"] = I_int_B4_2
    results["Simpson"]["I_int_B5"] = I_int_B5_2

    # 3. Numerical Integration
    prob = ODEProblem(odes!, [Value.S0, Value.E0, Value.I0, Value.R0, 0.0, 0.0, 0.0, 0.0, 0.0], (t[1], t[end]), Value.p_true)
    sol = DifferentialEquations.solve(prob, saveat=t, reltol=1e-14, abstol=1e-14)
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

function validation_plot(I, noise_level, t)
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

        push!(plot_list, p_val, p_err)
    end

    final_plot = plot(plot_list..., layout=(length(keys_to_check), 2),
                      size=(1000, 400 * length(keys_to_check)),
                      plot_title="Noise Level: $(noise_level*100)%")

    display(final_plot)
    return final_plot
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
        if plot_I_data
            plot!(p_err, t, I, label="I_data", color=:yellow)
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

        push!(plot_list, p_val, p_err, p_err_percentage)
    end

    final_plot = plot(plot_list..., layout=(length(keys_to_check), 3),
                      size=(1000, 400 * length(keys_to_check)),
                      plot_title="Noise Level: $(noise_level*100)%")

    display(final_plot)
    return final_plot
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    noise_levels = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    noise_level = [0]
    for noise in noise_level
        the_plot = validation_plot(I, noise, t)
        final_plot = validation_plot_complete(I, noise, t, plot_I_data=true)
        savefig(final_plot, "sanity_check_complete.pdf")
        savefig(the_plot, "sanity_check.pdf")
    end

    for noise in noise_levels
        validation_print(I, noise, t)
    end
end

main()