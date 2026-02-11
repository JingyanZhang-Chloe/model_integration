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
    const true_vals = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0]
    const scales = [0.00001, 0.01, 0.001, 10000, 0.1]

    const lb = [0.0, 0.0, 0.0, 0.0, 0.0]
    const ub = [Inf, Inf, Inf, Inf, Inf]

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

    function cumintegrate(x, y)
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

    function cumintegrate_simpson_uniform(x::AbstractVector, y::AbstractVector)
        n = length(x)
        T = promote_type(eltype(x), eltype(y))
        output = zeros(T, n)

        if n == 1
            error("cumintegrate requires at least 2 points")
        end

        if n == 2
            output[2] = (x[2] - x[1]) * (y[1] + y[2]) / 2
            return output
        end

        output[1] = zero(T)
        output[2] = (x[2] - x[1]) * (y[1] + y[2]) / 2

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

        if iseven(n)
            # Use the last 2 points
            h = x[n] - x[n-1]
            trap_step = h * (y[n-1] + y[n]) / 2
            output[n] = output[n-1] + (trap_step)
        end

        return output
    end

    function get_blocks(I_data::Vector{Float64}, t::Vector{Float64}, method::String)
        I0 = I_data[1]

        if method == "T"
            I_int = cumul_integrate(t, I_data)
            B1 = 1
            B2 = I_int
            B3 = cumul_integrate(t, I_data.^2 .- I0^2)
            B4 = t .* I_int .- cumul_integrate(t, t .* I_data)
            B5 = t .* cumul_integrate(t, I_data.^2) .- cumul_integrate(t, t .* (I_data.^2))
            B6 = cumul_integrate(t, (I_int).^2)
        elseif method == "S"
            I_int = cumintegrate(t, I_data)
            B1 = 1
            B2 = I_int
            B3 = cumintegrate(t, I_data.^2 .- I0^2)
            B4 = t .* I_int .- cumintegrate(t, t .* I_data)
            B5 = t .* cumintegrate(t, I_data.^2) .- cumintegrate(t, t .* (I_data.^2))
            B6 = cumintegrate(t, (I_int).^2)
        elseif method == "S_uniform"
            I_int = cumintegrate_simpson_uniform(t, I_data)
            B1 = 1
            B2 = I_int
            B3 = cumintegrate_simpson_uniform(t, I_data.^2 .- I0^2)
            B4 = t .* I_int .- cumintegrate_simpson_uniform(t, t .* I_data)
            B5 = t .* cumintegrate_simpson_uniform(t, I_data.^2) .- cumintegrate_simpson_uniform(t, t .* (I_data.^2))
            B6 = cumintegrate_simpson_uniform(t, (I_int).^2)
        else
            @error "method must be T or S"
        end

        return I0, B1, B2, B3, B4, B5, B6
    end

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

    function get_ideal_blocks(t::Vector{Float64})
        prob = ODEProblem(odes!, [Value.S0, Value.E0, Value.I0, Value.R0, 0.0, 0.0, 0.0, 0.0, 0.0], (t[1], t[end]), Value.p_true)
        sol = DifferentialEquations.solve(prob, saveat=t, reltol=1e-14, abstol=1e-14)
        sol_arr = Array(sol)
        I = sol_arr[3, :]
        I0 = I[1]
        I_int = sol_arr[5, :]
        I_int_sq = sol_arr[6, :]
        I_int_int_sq = sol_arr[7, :]
        I_int_B4 = sol_arr[8, :]
        I_int_B5 = sol_arr[9, :]

        B1 = 1
        B2 = I_int
        B3 = (I_int_sq - (I0^2) .* t)
        B4 = (t .* I_int .- I_int_B4)
        B5 = (t .* I_int_sq .- I_int_B5)
        B6 = I_int_int_sq

        return I0, B1, B2, B3, B4, B5, B6, I
    end

    function residual(paras, I0, B1, B2, B3, B4, B5, B6, t)
        α, σ, γ, S0, E0 = paras

        C1 = σ * (E0 + I0) .* t .* B1
        C2 = - (γ + σ) .* B2
        C3 = - 0.5 * α .* B3
        C4 = (α * σ * (S0 + E0 + I0) - σ * γ) .* B4
        C5 = - α * (γ + σ) .* B5
        C6 = - 0.5 * α * σ * γ .* B6

        return I0 .+ C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
    end

    function comp_I_hat(paras_scaled, I0, B1, B2, B3, B4, B5, B6, t)
        α_eff  = paras_scaled[1] * Value.scales[1]
        σ_eff  = paras_scaled[2] * Value.scales[2]
        γ_eff  = paras_scaled[3] * Value.scales[3]
        S0_eff = paras_scaled[4] * Value.scales[4]
        E0_eff = paras_scaled[5] * Value.scales[5]

        C1 = σ_eff * (E0_eff + I0) .* t .* B1
        C2 = - (γ_eff + σ_eff) .* B2
        C3 = - 0.5 * α_eff .* B3
        C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* B4
        C5 = - α_eff * (γ_eff + σ_eff) .* B5
        C6 = - 0.5 * α_eff * σ_eff * γ_eff .* B6

        return I0 .+ C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
    end

    function run_experiments(u0::Vector, I_data::Vector, t::Vector, method::String; scales=Value.scales)::NamedTuple
        alpha_list = Float64[]
        sigma_list = Float64[]
        gamma_list = Float64[]
        S0_list = Float64[]
        E0_list = Float64[]

        blocks = get_blocks(I_data, t, method)
        function model(x, p_normalized)
            p_real = p_normalized .* scales

            push!(alpha_list, p_real[1])
            push!(sigma_list, p_real[2])
            push!(gamma_list, p_real[3])
            push!(S0_list, p_real[4])
            push!(E0_list, p_real[5])

            return residual(p_real, blocks..., x)
        end

        u0_normalized = u0 ./ scales

        fit = curve_fit(model, t, I_data, u0_normalized, lower=Value.lb, upper=Value.ub)
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

    function get_error(est::Vector, true_value::Vector=Value.true_vals)::Vector
        return abs.(est .- true_value) ./ true_value .* 100
    end

    function get_RSS(est::Vector, true_value::Vector)::Float64
        return sum((est .- true_value).^2)
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

    function best_solution(solution_list::Vector{Vector{Float64}}, I_data::Vector, I0, B1, B2, B3, B4, B5, B6, t::Vector)
        best_sol = Float64[]
        best_err = Inf

        for param in solution_list
            I_hat = residual(param, I0, B1, B2, B3, B4, B5, B6, t)
            err = sum((I_hat .- I_data).^2)
            if err <= best_err
                best_err = err
                best_sol = param
            end
        end

        return best_sol, best_err
    end

    function best_solution_Trap(solution_list::Vector{Vector{Float64}}, I_data::Vector, t::Vector)
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

    function best_solution_Simp(solution_list::Vector{Vector{Float64}}, I_data::Vector, t::Vector)
        B = get_blocks_simpson(I_data, t)
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

    function RSS_I_data(I_data::Vector{Float64}, I::Vector{Float64})
        return sum((I_data .- I).^2)
    end

end