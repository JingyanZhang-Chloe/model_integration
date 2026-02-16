# seir_model.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 02/02/2026
=#

using Revise
using Random
include("SEIRModels.jl")
using .Logic
using .Value

"""
function main()
    t = collect(1.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t, plot=false)
    I_data = I .+ 0.01 .* I .* randn(length(I))
    u0 = [0.00003, 0.03, 0.003, 8000, 2]
    results = Logic.run_experiments(u0, I_data, t, "T")
    Logic.print_results(results)
end
"""

function main()

    steps = [10.0, 20.0, 25.0, 50.0, 100.0]
    # Time grid

    for step in steps
        t = collect(1.0:step:1000.0)

        # Generate fixed I_data (important: keep ONE realization)
        Random.seed!(1234)   # ensures reproducibility
        S, E, I, R = Logic.simulate_seir(t, plot=false)
        I_data = I .+ 0.01 .* I .* randn(length(I))

        # Baseline initial guess
        u0_baseline = [0.00003, 0.03, 0.003, 8000.0, 2.0]

        println("========================================")
        println("Baseline run with step size $step")
        println("========================================")

        results_base = Logic.run_experiments(u0_baseline, I_data, t, "T")
        Logic.print_results(results_base)
    end

    # OAT loop
    """
    param_names = String["α", "σ", "γ", "S0", "E0"]

    println("\n========================================")
    println("One-at-a-Time Sensitivity")
    println("========================================")

    for i in 1:length(u0_baseline)

        for factor in [0.1, 10.0]

            u0_test = copy(u0_baseline)
            u0_test[i] *= factor

            label = "$(param_names[i])_$(factor)x"

            println("\n----------------------------------------")
            println("Running case: $label")
            println("----------------------------------------")

            results = Logic.run_experiments(u0_test, I_data, t, "S")
            Logic.print_results(results)
        end
    end
    """
end

main()