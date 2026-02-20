# SEIRModels_ratio.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 12/02/2026
=#

module Value_R

    # True Value
    const S0 = 0.99
    const E0 = 0.0
    const I0 = 0.01
    const R0 = 0.0

    const α = 0.2
    const σ = 0.01
    const γ = 0.005

    const p_true = [α, σ, γ]
    const u = [S0, E0, I0, R0]
    const true_vals = [α, σ, γ, S0, E0]
    const scales = [0.01, 0.01, 0.01, 1.0, 1.0]

    const lb = [0.0, 0.0, 0.0, 0.8, 0.0] # Forcing S0 to be greater than 0.8
    const ub = [Inf, Inf, Inf, 1.0, 1.0]

    const T = 100.0
end


module Logic_R

    using ..Value_R
    using DifferentialEquations
    using Plots
    using LsqFit
    using NumericalIntegration
    using Statistics

    function seir!(du, u, p, t)
        S, E, I, R = u
        α, σ, γ = p
        du[1] = - α * S * I
        du[2] = α * S * I - σ * E
        du[3] = σ * E - γ * I
        du[4] = γ * I
    end

    function simulate_seir(t; u0=Value_R.u, p=Value_R.p_true, plot=false)
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

    function cumintegrate(x::AbstractVector, y::AbstractVector)
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
        prob = ODEProblem(odes!, [Value_R.S0, Value_R.E0, Value_R.I0, Value_R.R0, 0.0, 0.0, 0.0, 0.0, 0.0], (t[1], t[end]), Value_R.p_true)
        sol = DifferentialEquations.solve(prob, saveat=t, reltol=1e-16, abstol=1e-16)
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
        α_eff  = paras_scaled[1] * Value_R.scales[1]
        σ_eff  = paras_scaled[2] * Value_R.scales[2]
        γ_eff  = paras_scaled[3] * Value_R.scales[3]
        S0_eff = paras_scaled[4] * Value_R.scales[4]
        E0_eff = paras_scaled[5] * Value_R.scales[5]

        C1 = σ_eff * (E0_eff + I0) .* t .* B1
        C2 = - (γ_eff + σ_eff) .* B2
        C3 = - 0.5 * α_eff .* B3
        C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* B4
        C5 = - α_eff * (γ_eff + σ_eff) .* B5
        C6 = - 0.5 * α_eff * σ_eff * γ_eff .* B6

        return I0 .+ C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
    end

    function run_experiments(u0::Vector, I_data::Vector, t::Vector, method::String; I::Union{Nothing, Vector{Float64}}=nothing, scales=Value_R.scales)::NamedTuple
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

        fit = curve_fit(model, t, I_data, u0_normalized, lower=Value_R.lb ./ scales, upper=Value_R.ub ./ scales)
        p_hat = fit.param .* scales

        return (
            α_trace = alpha_list,
            σ_trace = sigma_list,
            γ_trace = gamma_list,
            S0_trace = S0_list,
            E0_trace = E0_list,
            estimated = p_hat,
            true_params = [Value_R.α, Value_R.σ, Value_R.γ, Value_R.S0, Value_R.E0],
            t = t,
            initial_guesses = u0,
            I_data = I_data,
            I = I,
            blocks = blocks
        )
    end

    function run_experiments_k_points(
        u0::Vector{Float64},
        k_points::Vector{Int},
        noise::Float64,
        I_true::Vector{Float64},
        t::Vector{Float64};
        scales = Value_R.scales,
        num_of_iters::Int = 20
    )
        """
        u0: initial guesses
        k_points: list of k where k means first k points
        """
        params  = String["alpha", "sigma", "gamma", "S0", "E0"]
        methods = ["T", "S"]

        # Store samples: re[k][method]["p_hat"] = Vector{Vector{Float64}}
        #                re[k][method]["rss"]   = Vector{Float64}
        #                re[k][method]["err"]   = Vector{Vector{Float64}}
        # Notice that we store results from every iteration
        re = Dict{Int, Dict{String, Dict{String, Any}}}()
        for k in k_points
            re[k] = Dict{String, Dict{String, Any}}()
            for m in methods
                re[k][m] = Dict{String, Any}()
                re[k][m]["p_hat"] = Vector{Vector{Float64}}()
                re[k][m]["rss"]   = Float64[]
                re[k][m]["err"]   = Vector{Vector{Float64}}()
            end
        end

        u0_norm = u0 ./ scales
        lb_norm = Value_R.lb ./ scales
        ub_norm = Value_R.ub ./ scales

        # Baseline: rss of I_data vs true I_true, averaged across iters
        rss_baseline_list = Float64[]

        for iter in 1:num_of_iters
            I_data = I_true .+ noise .* I_true .* randn(length(I_true))
            push!(rss_baseline_list, get_RSS(I_data, I_true))

            for k in k_points
                t_k = t[1:k]
                I_k = I_data[1:k]

                for m in methods
                    blocks_k = get_blocks(I_k, t_k, m)

                    model(x, p_norm) = begin
                        p_real = p_norm .* scales
                        residual(p_real, blocks_k..., x)
                    end

                    fit = curve_fit(model, t_k, I_k, u0_norm; lower=lb_norm, upper=ub_norm)
                    p_hat = fit.param .* scales

                    # Predict full period I_hat using FULL blocks built from full data
                    blocks_full = get_blocks(I_data, t, m)
                    I_hat_full  = residual(p_hat, blocks_full..., t)

                    rss_pred = get_RSS(I_hat_full, I_true)
                    err_pct  = get_error(p_hat, Value_R.true_vals)

                    push!(re[k][m]["p_hat"], p_hat)
                    push!(re[k][m]["rss"], rss_pred)
                    push!(re[k][m]["err"], err_pct)
                end
            end
        end

        rss_baseline_mean = mean(rss_baseline_list)
        rss_baseline_std  = std(rss_baseline_list)

        println("\n================ Early-Time Data Truncation ================")
        println("Noise level: $noise")
        println("Baseline RSS (I_data vs I_true): mean=$(rss_baseline_mean), std=$(rss_baseline_std)")
        println("------------------------------------------------------------")

        for k in k_points
            println("\n>>> k = $k")

            for m in methods
                label = (m == "T" ? "Trap (T)" : "Simpson (S)")

                p_samples = re[k][m]["p_hat"]::Vector{Vector{Float64}}
                rss_list  = re[k][m]["rss"]::Vector{Float64}
                err_list  = re[k][m]["err"]::Vector{Vector{Float64}}

                P = hcat(p_samples...)   # 5 x num_iters
                E = hcat(err_list...)    # 5 x num_iters

                p_mean   = vec(mean(P; dims=2))
                p_std    = vec(std(P;  dims=2))
                err_mean = vec(mean(E; dims=2))
                err_std  = vec(std(E;  dims=2))

                rss_mean = mean(rss_list)
                rss_std  = std(rss_list)

                println("  Method: $label")
                println("  RSS (sum((I_hat .- I_data).^2)): mean=$(rss_mean), std=$(rss_std)")
                println("  RSS ratio to baseline (mean): $(rss_mean / rss_baseline_mean)")

                for i in 1:5
                    println("    $(params[i]): mean=$(p_mean[i]) std=$(p_std[i]) | err% mean=$(err_mean[i]) std=$(err_std[i])")
                end
            end
        end

        println("\n============================================================\n")
        return re
    end

    function get_error(est::Vector, true_value::Vector=Value_R.true_vals)::Vector
        return abs.(est .- true_value) ./ true_value .* 100
    end

    function get_RSS(est::Vector, true_value::Vector)::Float64
        return sum((est .- true_value).^2)
    end

    function print_results(results::NamedTuple; t=collect(1.0:10.0:1000.0))
        true_list = results.true_params
        estimated_list = results.estimated
        initial_list = results.initial_guesses
        err_list = get_error(estimated_list, true_list)

        cost = sum((true_list .- estimated_list).^2)
        iteration = length(results.α_trace)

        rss = nothing
        if results.I !== nothing
            I_hat = residual(estimated_list, results.blocks..., t)
            rss = get_RSS(I_hat, results.I) # RSS (sum((I_hat .- I_data).^2))
        end

        println("=" ^ 82)
        println("Estimation Results")
        println("Total cost: $cost | iteration steps: $iteration | RSS (sum((I_hat .- I_data).^2)): $rss")
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

    function RSS_I_data(I_data::Vector{Float64}, I::Vector{Float64})
        return sum((I_data .- I).^2)
    end

    function swap_project_SE0_for_sigma_gamma!(x::Vector{Float64}, I0::Float64)
        α, σ, γ, S0, E0 = x
        r = σ / γ

        σ_new = γ
        γ_new = σ

        # preserve σ(E0+I0)
        E0_new = r * (E0 + I0) - I0

        # preserve σ(S0+E0+I0)
        S0_new = r * S0

        x[2] = σ_new
        x[3] = γ_new
        x[4] = S0_new
        x[5] = E0_new
        return x
    end

    function project_S0E0_euclidean!(x::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})
        s0 = x[4]
        e0 = x[5]

        if s0 + e0 <= 1.0
            x[4] = s0
            x[5] = e0
            return x
        end

        # Feasible s must satisfy:
        #   lbS <= s <= ubS
        #   lbE <= 1-s <= ubE  =>  1-ubE <= s <= 1-lbE
        s_min = max(lb[4], 1.0 - ub[5])
        s_max = min(ub[4], 1.0 - lb[5])

        # Euclidean projection (smallest adjustment in L2)
        s_star = 0.5 * (s0 + 1.0 - e0)
        s_proj = clamp(s_star, s_min, s_max)
        e_proj = 1.0 - s_proj

        x[4] = s_proj
        x[5] = e_proj
        return x
    end

    function project_to_bounds(result::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, I0::Float64)::Vector{Float64}
        """
        Here bounds we are applying is
        all(lb_scaled .<= res .<= ub_scaled) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
        """
        x = copy(result)

        if x[2] < x[3]
            swap_project_SE0_for_sigma_gamma!(x, I0)
        end

        x = clamp.(x, lb, ub)

        if lb[4] + lb[5] > 1
            @warn "Infeasible bounds: lb[4] + lb[5] > 1, cannot satisfy S0 + E0 <= 1"
            return x
        end

        if x[4] + x[5] > 1
            project_S0E0_euclidean!(x, lb, ub)
        end

        x = clamp.(x, lb, ub)

        return x
    end

    function noise_level_analysis(I, t, noise_levels, x_noise_percent, x0, method; num_of_iters=20)
        params = String["alpha", "sigma", "gamma", "S0", "E0"]
        results = Dict{String, Dict{String, Vector{Float64}}}()
        for p in params
            results[p] = Dict("mean" => Float64[], "std" => Float64[])
        end

        for noise in noise_levels
            param_samples = Dict{String, Vector{Float64}}()
            for p in params
                param_samples[p] = Float64[]
            end

            for _ in 1:num_of_iters
                I_data = I .+ noise .* I .* randn(length(I))
                res = run_experiments(x0, I_data, t, method).estimated

                push!(param_samples["alpha"], res[1])
                push!(param_samples["sigma"], res[2])
                push!(param_samples["gamma"], res[3])
                push!(param_samples["S0"],    res[4])
                push!(param_samples["E0"],    res[5])
            end

            for p in keys(results)
                push!(results[p]["mean"], mean(param_samples[p]))
                push!(results[p]["std"],  std(param_samples[p]))
            end
        end

        # Plot 5 subplots
        true_values = Value_R.true_vals
        plt_list = Plots.Plot[]
        for (i, p) in enumerate(params)
            means = results[p]["mean"]
            stds  = results[p]["std"]

            lower = means .- stds
            upper = means .+ stds

            pl = plot(x_noise_percent, means;
                      label="Estimated $p",
                      marker=:circle,
                      markersize=3,
                      xlabel="Noise Level (% of max I)",
                      ylabel="Parameter Value",
                      title="Robustness of $p",
                      gridalpha=0.3)

            plot!(pl, x_noise_percent, lower; label="Mean ± Std", fillrange=upper, fillalpha=0.2, linealpha=0.0)

            hline!(pl, [true_values[i]]; label="True $p", linestyle=:dash, linewidth=2)

            push!(plt_list, pl)
        end

        final_plot = plot(plt_list...; layout=(1, 5), size=(1600, 350), background_color=:white)
        savefig(final_plot, "noise_level_analysis.pdf")
    end

    function num_of_datapoints_analysis(num_of_datapoints::Vector{Int}, noise, x0, method; num_of_iters=20)
        params = String["alpha", "sigma", "gamma", "S0", "E0"]
        results = Dict{String, Dict{String, Vector{Float64}}}()
        for p in params
            results[p] = Dict("mean" => Float64[], "std" => Float64[])
        end

        for ns in num_of_datapoints
            t = collect(range(0.0, 1000.0, length=ns))
            S, E, I, R = simulate_seir(t)
            param_samples = Dict{String, Vector{Float64}}()
            for p in params
                param_samples[p] = Float64[]
            end

            for _ in 1:num_of_iters
                I_data = I .+ noise .* I .* randn(length(I))
                res = run_experiments(x0, I_data, t, method).estimated

                push!(param_samples["alpha"], res[1])
                push!(param_samples["sigma"], res[2])
                push!(param_samples["gamma"], res[3])
                push!(param_samples["S0"],    res[4])
                push!(param_samples["E0"],    res[5])
            end

            for p in keys(results)
                push!(results[p]["mean"], mean(param_samples[p]))
                push!(results[p]["std"],  std(param_samples[p]))
            end
        end

        # Plot 5 subplots
        true_values = Value_R.true_vals
        plt_list = Plots.Plot[]
        for (i, p) in enumerate(params)
            means = results[p]["mean"]
            stds  = results[p]["std"]

            lower = means .- stds
            upper = means .+ stds

            pl = plot(num_of_datapoints, means;
                      label="Estimated $p",
                      marker=:circle,
                      markersize=3,
                      xlabel="Number of Data Points",
                      ylabel="Parameter Value",
                      title="Robustness of $p",
                      gridalpha=0.3)

            plot!(pl, num_of_datapoints, lower; label="Mean ± Std", fillrange=upper, fillalpha=0.2, linealpha=0.0)

            hline!(pl, [true_values[i]]; label="True $p", linestyle=:dash, linewidth=2)

            push!(plt_list, pl)
        end

        final_plot = plot(plt_list...; layout=(1, 5), size=(1600, 350), background_color=:white)
        savefig(final_plot, "num_of_datapoints_analysis.pdf")
    end

    function noise_level_analysis(I, t, noise_levels, x_noise_percent, x0; num_of_iters=20)
        params  = String["alpha", "sigma", "gamma", "S0", "E0"]
        methods = String["T", "S"]   # row1: T row2: S

        # results[method][param]["mean"/"std"] -> Vector over noise_levels
        results = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()
        for m in methods
            results[m] = Dict{String, Dict{String, Vector{Float64}}}()
            for p in params
                results[m][p] = Dict("mean" => Float64[], "std" => Float64[])
            end
        end

        for noise in noise_levels
            for m in methods
                param_samples = Dict(p => Float64[] for p in params)

                for _ in 1:num_of_iters
                    I_data = I .+ noise .* I .* randn(length(I))
                    res = run_experiments(x0, I_data, t, m).estimated

                    push!(param_samples["alpha"], res[1])
                    push!(param_samples["sigma"], res[2])
                    push!(param_samples["gamma"], res[3])
                    push!(param_samples["S0"],    res[4])
                    push!(param_samples["E0"],    res[5])
                end

                for p in params
                    push!(results[m][p]["mean"], mean(param_samples[p]))
                    push!(results[m][p]["std"],  std(param_samples[p]))
                end
            end
        end

        true_values = Value_R.true_vals

        # Compute ylims per parameter across BOTH methods (using mean ± std plus true)
        ylims_by_param = Dict{String, Tuple{Float64, Float64}}()
        for (i, p) in enumerate(params)
            vals = Float64[]
            for m in methods
                means = results[m][p]["mean"]
                stds  = results[m][p]["std"]
                append!(vals, means .- stds)
                append!(vals, means .+ stds)
            end
            push!(vals, true_values[i])

            ymin, ymax = minimum(vals), maximum(vals)
            pad = 0.05 * (ymax - ymin + eps())
            ylims_by_param[p] = (ymin - pad, ymax + pad)
        end

        # Build 2x5 plots
        plt_list = Plots.Plot[]
        for (row, m) in enumerate(methods)
            for (i, p) in enumerate(params)
                means = results[m][p]["mean"]
                stds  = results[m][p]["std"]
                lower = means .- stds
                upper = means .+ stds

                method_label = (m == "T" ? "Trap (T)" : "Simpson (S)")

                pl = plot(
                    x_noise_percent, means;
                    label="Estimated $p",
                    marker=:circle,
                    markersize=3,
                    xlabel=(row == 2 ? "Noise Level (% of max I)" : ""),  # only bottom row shows x label
                    ylabel="Parameter Value",
                    title="$p • $method_label",
                    gridalpha=0.3,
                    ylims=ylims_by_param[p],
                )

                plot!(pl, x_noise_percent, lower; label="Mean ± Std", fillrange=upper, fillalpha=0.2, linealpha=0.0)
                hline!(pl, [true_values[i]]; label="True $p", linestyle=:dash, linewidth=2)

                push!(plt_list, pl)
            end
        end

        final_plot = plot(plt_list...; layout=(2, 5), size=(1600, 650), background_color=:white)
        savefig(final_plot, "noise_level_analysis.pdf")
    end

    function num_of_datapoints_analysis(num_of_datapoints::Vector{Int}, noise, x0; num_of_iters=20)
        params  = String["alpha", "sigma", "gamma", "S0", "E0"]
        methods = String["T", "S"]

        results = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()
        for m in methods
            results[m] = Dict{String, Dict{String, Vector{Float64}}}()
            for p in params
                results[m][p] = Dict("mean" => Float64[], "std" => Float64[])
            end
        end

        for ns in num_of_datapoints
            t = collect(range(0.0, 1000.0, length=ns))
            S, E, I, R = simulate_seir(t)

            for m in methods
                param_samples = Dict(p => Float64[] for p in params)

                for _ in 1:num_of_iters
                    I_data = I .+ noise .* I .* randn(length(I))
                    res = run_experiments(x0, I_data, t, m).estimated

                    push!(param_samples["alpha"], res[1])
                    push!(param_samples["sigma"], res[2])
                    push!(param_samples["gamma"], res[3])
                    push!(param_samples["S0"],    res[4])
                    push!(param_samples["E0"],    res[5])
                end

                for p in params
                    push!(results[m][p]["mean"], mean(param_samples[p]))
                    push!(results[m][p]["std"],  std(param_samples[p]))
                end
            end
        end

        true_values = Value_R.true_vals

        ylims_by_param = Dict{String, Tuple{Float64, Float64}}()
        for (i, p) in enumerate(params)
            vals = Float64[]
            for m in methods
                means = results[m][p]["mean"]
                stds  = results[m][p]["std"]
                append!(vals, means .- stds)
                append!(vals, means .+ stds)
            end
            push!(vals, true_values[i])

            ymin, ymax = minimum(vals), maximum(vals)
            pad = 0.05 * (ymax - ymin + eps())
            ylims_by_param[p] = (ymin - pad, ymax + pad)
        end

        plt_list = Plots.Plot[]
        for (row, m) in enumerate(methods)
            for (i, p) in enumerate(params)
                means = results[m][p]["mean"]
                stds  = results[m][p]["std"]
                lower = means .- stds
                upper = means .+ stds

                method_label = (m == "T" ? "Trap (T)" : "Simpson (S)")

                pl = plot(
                    num_of_datapoints, means;
                    label="Estimated $p",
                    marker=:circle,
                    markersize=3,
                    xlabel=(row == 2 ? "Number of Data Points" : ""),
                    ylabel="Parameter Value",
                    title="$p • $method_label",
                    gridalpha=0.3,
                    ylims=ylims_by_param[p],
                )

                plot!(pl, num_of_datapoints, lower; label="Mean ± Std", fillrange=upper, fillalpha=0.2, linealpha=0.0)
                hline!(pl, [true_values[i]]; label="True $p", linestyle=:dash, linewidth=2)

                push!(plt_list, pl)
            end
        end

        final_plot = plot(plt_list...; layout=(2, 5), size=(1600, 650), background_color=:white)
        savefig(final_plot, "num_of_datapoints_analysis.pdf")
    end

end
