# homotopy_continuation_promax.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 10/02/2026
=#

using Revise
using HomotopyContinuation, DynamicPolynomials, Random, Printf
include("SEIRModels.jl")
using .Logic
using .Value

@var α, σ, γ, S0, E0
const variables_scaled = [α, σ, γ, S0, E0]

function HC_least_squares(t::Vector{Float64}, I_data::Vector{Float64}, vars_scaled::Vector, method::String)
    B = Logic.get_blocks(I_data, t, method)
    I_hat = Logic.comp_I_hat(vars_scaled, B..., t)
    J = sum((I_hat .- I_data).^2) / length(I_data)
    system_eqs = differentiate(J, vars_scaled)
    C = System(system_eqs, variables=vars_scaled)

    result_scaled = HomotopyContinuation.solve(C;
        start_system = :polyhedral,
        tracker_options = TrackerOptions(automatic_differentiation=3),
    )

    real_results_scaled = real_solutions(result_scaled)
    real_results = [result .* Value.scales for result in real_results_scaled]

    filtered_results = filter(real_results) do res
        all(Value.lb .<= res .<= Value.ub) && (res[2] > res[3])
    end

    if isempty(filtered_results)
        @error "No physical solutions found by HC, returning un-filtered real results"
        return real_results
    end

    best_init, HC_RSS = Logic.best_solution(filtered_results, I_data, B..., t)

    model(t_para, p_scaled) = Logic.comp_I_hat(p_scaled, B..., t_para)

    u0 = best_init ./ Value.scales
    lb_scaled = Value.lb ./ Value.scales
    ub_scaled = Value.ub ./ Value.scales

    fit = curve_fit(model, t, I_data, u0; lower=lb_scaled, upper=ub_scaled, show_trace=true,
        x_tol=1e-14,
        g_tol=1e-14,
        autodiff=:forward,
    )

    final_result = fit.param .* Value.scales

    return (
        estimated = final_result,
        true_params = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0],
        fit_obj = fit,
        initial_guesses = best_init,
        HC_RSS = HC_RSS,
        t = t,
        I_data = I_data
    )
end

function print_HC_least_squares(results)
    est = results.estimated
    true_p = results.true_params
    init_p = results.initial_guesses
    fit = results.fit_obj

    final_RSS = sum(fit.resid.^2)
    hc_RSS = results.HC_RSS

    println("="^110)
    println("                                SEIR MODEL HIGH-PRECISION RESULTS")
    println("="^110)

    @printf("%-10s | %-15s | %-15s | %-15s | %-12s\n",
            "Param", "True Value", "HC Initial", "Final Fit", "Rel. Error (%)")
    println("-"^110)

    # Parameters
    names = String["alpha", "sigma", "gamma", "S0", "E0"]
    for i in 1:length(names)
        rel_err = abs(est[i] - true_p[i]) / true_p[i] * 100
        @printf("%-10s | %1.16e | %1.16e | %1.16e | %1.4e\n",
                names[i], true_p[i], init_p[i], est[i], rel_err)
    end
    println("-"^110)

    # Performance Metrics
    improvement = hc_RSS - final_RSS
    improvement_pct = (improvement / hc_RSS) * 100

    println("\n[Residual Sum of Squares (RSS) Analysis]")
    @printf("  Initial RSS (HC):      %1.16e\n", hc_RSS)
    @printf("  Final RSS (LsqFit):    %1.16e\n", final_RSS)
    @printf("  Absolute Δ RSS:        %1.16e\n", improvement)
    @printf("  Improvement Pct:       %1.16e%%\n", improvement_pct)

    # 3. Solver Metadata
    println("\n[Solver Metadata]")
    println("  Converged:             $(fit.converged)")
    println("="^110)
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t, plot=false)
    I_data = I .+ 0.01 .* I .* randn(length(I))

    results = HC_least_squares(t, I_data, variables_scaled, "S")
    print_HC_least_squares(results)
end

main()
