# sanity_check_blocks.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 11/02/2026
=#

include("SEIRModels.jl")
using .Logic
using .Value

function blocks_check(I_data, t)::Dict{String, Dict{String, Vector{Float64}}}
    results = Dict{String, Dict{String, Vector{Float64}}}()

    results["Trapezoidal"] = Dict{String, Vector{Float64}}()
    results["Simpson"] = Dict{String, Vector{Float64}}()
    results["Baseline"] = Dict{String, Vector{Float64}}()

    # Trapzoidal
    I0_T, B1_T, B2_T, B3_T, B4_T, B5_T, B6_T = Logic.get_blocks(I_data, t, "T")
    results["Trapezoidal"]["B2"] = B2_T
    results["Trapezoidal"]["B3"] = B3_T
    results["Trapezoidal"]["B4"] = B4_T
    results["Trapezoidal"]["B5"] = B5_T
    results["Trapezoidal"]["B6"] = B6_T

    # Simpson
    I0_S, B1_S, B2_S, B3_S, B4_S, B5_S, B6_S = Logic.get_blocks(I_data, t, "S")
    results["Simpson"]["B2"] = B2_S
    results["Simpson"]["B3"] = B3_S
    results["Simpson"]["B4"] = B4_S
    results["Simpson"]["B5"] = B5_S
    results["Simpson"]["B6"] = B6_S

    # Baseline
    I0, B1, B2, B3, B4, B5, B6, I = Logic.get_ideal_blocks(t)
    results["Baseline"]["B2"] = B2
    results["Baseline"]["B3"] = B3
    results["Baseline"]["B4"] = B4
    results["Baseline"]["B5"] = B5
    results["Baseline"]["B6"] = B6

    return results
end

function print_block_comparison(results::Dict{String, Dict{String, Vector{Float64}}})
    methods = ["Trapezoidal", "Simpson"]
    blocks = String["B2", "B3", "B4", "B5", "B6"]

    baseline = results["Baseline"]

    println("="^95)
    println(" BLOCK VALIDATION CHECK ")
    println("="^95)
    println(rpad("Block", 10),
            rpad("Max Err (T)", 15),
            rpad("Max Err (S)", 15),
            rpad("Sum Err (T)", 15),
            rpad("Sum Err (S)", 15),
            "Gain (T/S)")
    println("-"^95)

    for blk in blocks
        base = baseline[blk]
        trap = results["Trapezoidal"][blk]
        simp = results["Simpson"][blk]

        err_T = abs.(trap .- base)
        err_S = abs.(simp .- base)

        max_T = maximum(err_T)
        max_S = maximum(err_S)

        sum_T = sum(err_T)
        sum_S = sum(err_S)

        gain = sum_T / sum_S

        println(rpad(blk, 10),
                @sprintf("%-14.4e", max_T),
                @sprintf("%-14.4e", max_S),
                @sprintf("%-14.4e", sum_T),
                @sprintf("%-14.4e", sum_S),
                @sprintf("%.2f x", gain))
    end

    println("="^95)
end

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    I_data = I .+ 0 .* I .* randn(length(I))
    results = blocks_check(I_data, t)
    print_block_comparison(results)
end

main()
