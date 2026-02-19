# seir_model_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

include("SEIRModels_ratio.jl")
using .Logic_R
using .Value_R

function clamp_to_bounds!(x, lb, ub; eps=1e-12)
    for k in eachindex(x)
        lo = lb[k]
        hi = ub[k]
        # if hi is Inf, skip upper clamp
        if isfinite(hi)
            x[k] = min(max(x[k], lo + eps), hi - eps)
        else
            x[k] = max(x[k], lo + eps)
        end
    end
    return x
end

function main()
    t = collect(1.0:10.0:1000.0)
    S, E, I, R = Logic_R.simulate_seir(t)
    noise = 0.01
    I_data = I .+ noise .* I .* randn(length(I))

    k_points = [10 * i for i in 1:1:10]

    # Baseline initial guess
    u0_baseline = [0.3, 0.03, 0.003, 0.8, 0.02]

    # Logic_R.run_experiments_k_points(u0_baseline, k_points, noise, I, t)

    #=
    println("\n----------------------------------------")
    println("Running case: $noise, t: $t")
    println("----------------------------------------")

    println("Simpson")
    results_S = Logic_R.run_experiments(u0_baseline, I_data, t, "S"; I=I)
    Logic_R.print_results(results_S)

    println("Trapezoidal")
    results_T = Logic_R.run_experiments(u0_baseline, I_data, t, "T"; I=I)
    Logic_R.print_results(results_T)
    =#

    #=
    noise_levels = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

    for noise in noise_levels
        I_daaaa = I .+ noise .* I .* randn(length(I))

        println("\n----------------------------------------")
        println("Running case: $noise")
        println("----------------------------------------")

        println("Simpson")
        results_S = Logic_R.run_experiments(u0_baseline, I_daaaa, t, "S")
        Logic_R.print_results(results_S)

        println("Trapezoidal")
        results_T = Logic_R.run_experiments(u0_baseline, I_daaaa, t, "T")
        Logic_R.print_results(results_T)
    end
    =#

    noise_steps = 41
    noise_levels = [0.005 * i for i in 0:noise_steps-1]      # 0 to 0.2
    x_noise_percent = [0.5 * i for i in 0:noise_steps-1]     # 0% to 20%
    # Logic_R.noise_level_analysis(I, t, noise_levels, x_noise_percent, u0_baseline)

    num_of_datapoints = [i for i in 10:5:100]
    # Logic_R.num_of_datapoints_analysis(num_of_datapoints, 0.01, u0_baseline)

    println()
end

main()