
"""
Example for NMF (Non-negative Matrix Factorization)
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
using NMF

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Dimension
    D = 2
    M = 4
    N = 2

    # Hyper-parameters
    a_w = rand(Normal(), (D, M))
    a_h = rand(Normal(), (M, N))
    b_w = rand(Normal(), M)
    b_h = rand(Normal(), M)
    
    # Prior
    prior = NMF.NMFModel(D, M, N, a_w, a_h, b_w, b_h)

    # Sample dummy data
    X, S, W, H = NMF.sample_data(20, prior)

    # Inference
    max_iter = 2
    # posterior, X = NMF.VI(Y, prior, max_iter)
end

# -----------------------------------------------------------
# Main function
# -----------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "func"
            help = "selected function"
            arg_type = String
            default = "dummy"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    # Function hash table
    func_dict = Dict([("dummy", check_with_dummy_data),])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()
