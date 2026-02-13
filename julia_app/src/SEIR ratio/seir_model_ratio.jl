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

    Random.seed!(1234)
    S, E, I, R = Logic_R.simulate_seir(t)
    I_data = I .+ 0.01 .* I .* randn(length(I))

    # Baseline initial guess
    u0_baseline = [0.3, 0.03, 0.003, 0.8, 0.02]

    param_names = String["α", "σ", "γ", "S0", "E0"]


    println("========================================")
    println("Baseline run")
    println("========================================")

    results_base = Logic_R.run_experiments(u0_baseline, I_data, t, "T")
    Logic_R.print_results(results_base)

    println("\n========================================")
    println("One-at-a-Time Sensitivity")
    println("========================================")

    # OAT loop
    for i in 1:length(u0_baseline)

        for factor in [0.1, 10.0]

            u0_test = copy(u0_baseline)
            u0_test[i] *= factor
            clamp_to_bounds!(u0_test, Value_R.lb, Value_R.ub)

            label = "$(param_names[i])_$(factor)x"

            println("\n----------------------------------------")
            println("Running case: $label")
            println("----------------------------------------")

            results = Logic_R.run_experiments(u0_test, I_data, t, "T")
            Logic_R.print_results(results)
        end
    end
end

main()