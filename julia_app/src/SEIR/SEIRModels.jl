# SEIRModels.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 03/02/2026
=#

using Revise
using DifferentialEquations
using Plots
using LsqFit
using NumericalIntegration

module Value

    # True Value
    const S0 = 9999.0
    const E0 = 0.0
    const I0 = 1.0
    const R0 = 0.0

    const α = 0.00002
    const σ = 0.01
    const γ = 0.005

    const p_true = [α, σ, γ]
    const u = [S0, E0, I0, R0]
    const scales = [0.00001, 0.01, 0.001, 10000, 0.1]

end


module Logic

    using ..Value
    using DifferentialEquations
    using Plots
    using LsqFit
    using NumericalIntegration

    function seir!(du, u, p, t)
        S, E, I, R = u
        α, σ, γ = p
        du[1] = - α * S * I
        du[2] = α * S * I - σ * E
        du[3] = σ * E - γ * I
        du[4] = γ * I
    end

    function simulate_seir(t; u0=Value.u, p=Value.p_true, plot=false)
        prob = ODEProblem(seir!, u0, (t[1], t[end]), p)
        sol = DifferentialEquations.solve(prob, saveat=t)
        sol_arr = Array(sol)
        S = sol_arr[1, :]
        E = sol_arr[2, :]
        I = sol_arr[3, :]
        R = sol_arr[4, :]

        if plot
            data_to_plot = hcat(S, E, I, R)
            println("Plotting data of size: ", size(data_to_plot))
            plt = Plots.plot(t, data_to_plot,
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

    function get_blocks(I_data, t)
        I0 = I_data[1]
        I_int = cumul_integrate(t, I_data)

        B1 = 1
        B2 = I_int
        B3 = cumul_integrate(t, I_data.^2 .- I0^2)
        B4 = t .* I_int .- cumul_integrate(t, t .* I_data)
        B5 = t .* cumul_integrate(t, I_data.^2) .- cumul_integrate(t, t .* (I_data.^2))
        B6 = cumul_integrate(t, (I_int).^2)

        return B1, B2, B3, B4, B5, B6
    end

    function residual(paras, I_data, B1, B2, B3, B4, B5, B6, t)
        α, σ, γ, S0, E0 = paras
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

    function run_experiments(u0::Vector, I_data::Vector, t::Vector; scales=Value.scales)::NamedTuple
        alpha_list = Float64[]
        sigma_list = Float64[]
        gamma_list = Float64[]
        S0_list = Float64[]
        E0_list = Float64[]

        B1, B2, B3, B4, B5, B6 = get_blocks(I_data, t)
        function model(x, p_normalized)
            p_real = p_normalized .* scales

            push!(alpha_list, p_real[1])
            push!(sigma_list, p_real[2])
            push!(gamma_list, p_real[3])
            push!(S0_list, p_real[4])
            push!(E0_list, p_real[5])

            return residual(p_real, I_data, B1, B2, B3, B4, B5, B6, x)
        end

        lb = [0.0, 0.0, 0.0, 0.0, 0.0]
        ub = [Inf, Inf, Inf, Inf, Inf]
        u0_normalized = u0 ./ scales

        fit = curve_fit(model, t, I_data, u0_normalized, lower=lb, upper=ub)
        p_hat = fit.param .* scales

        return (
            α_trace = alpha_list,
            σ_trace = sigma_list,
            γ_trace = gamma_list,
            S0_trace = S0_list,
            E0_trace = E0_list,
            estimated = p_hat,
            true_params = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0],
            t = t,
            initial_guesses = u0,
            I_data = I_data
        )
    end

    function get_error(est::Vector, true_value::Vector)::Vector
        return abs.(est .- true_value) ./ true_value .* 100
    end

    function print_results(results::NamedTuple)
        true_list = results.true_params
        estimated_list = results.estimated
        initial_list = results.initial_guesses
        err_list = get_error(estimated_list, true_list)

        cost = sum((true_list .- estimated_list).^2)
        iteration = length(results.α_trace)

        println("=" ^ 82)
        println("Estimation Results")
        println("Total cost: $cost | iteration steps: $iteration")
        println("=" ^ 82)

        param_labels = String["α", "σ", "γ", "S0", "E0"]

        for (label, t_val, i_val, e_val, err_val) in zip(param_labels, true_list, initial_list, estimated_list, err_list)
            println("----- $label -----")
            println("True: $t_val | Guess: $i_val | Result: $e_val | Error: $err_val")
        end
    end

    function plot_results(results::NamedTuple)
        p1 = plot(results.α_trace, title="Alpha Convergence", color=:orange, label="Est Alpha", m=:o, ms=3)
        hline!([results.true_params[1]], label="True", color=:black, ls=:dash)

        p2 = plot(results.σ_trace, title="Sigma Convergence", color=:purple, label="Est Sigma", m=:o, ms=3)
        hline!([results.true_params[2]], label="True", color=:black, ls=:dash)

        p3 = plot(results.γ_trace, title="Gamma Convergence", color=:green, label="Est Gamma", m=:o, ms=3)
        hline!([results.true_params[3]], label="True", color=:black, ls=:dash)

        p4 = plot(results.S0_trace, title="S0 Convergence", color=:blue, label="Est S0", m=:o, ms=3)
        hline!([results.true_params[4]], label="True", color=:black, ls=:dash)

        p5 = plot(results.E0_trace, title="E0 Convergence", color=:red, label="Est E0", m=:o, ms=3)
        hline!([results.true_params[5]], label="True", color=:black, ls=:dash)

        final_plot = plot(p1, p2, p3, p4, p5, layout=(1, 5), size=(1600, 300))

        display(final_plot)
    end

    function best_solution(solution_list::Vector{Vector{Float64}}, I_data::Vector, t::Vector)
        B = get_blocks(I_data, t)
        best_sol = Float64[]
        best_err = Inf

        for param in solution_list
            if any(param .<= 0)
                continue
            end

            I_hat = residual(param, I_data, B..., t)
            err = sum((I_hat .- I_data).^2)
            if err <= best_err
                best_err = err
                best_sol = param
            end
        end

        return best_sol, best_err
    end


end