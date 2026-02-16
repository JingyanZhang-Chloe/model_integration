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

    # Random.seed!(1234)
    S, E, I, R = Logic_R.simulate_seir(t)
    noise_levels = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    # Baseline initial guess
    u0_baseline = [0.3, 0.03, 0.003, 0.8, 0.02]

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
end

main()