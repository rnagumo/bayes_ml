
"""
Example for NMF (Non-negative Matrix Factorization)
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
import PoissonHMM

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Hyper-parameters
    K = 2
    a = rand(K)
    b = rand(K)
    alpha = rand(K)
    beta = rand(K, K)
    
    # Prior
    prior = PoissonHMM.PoissonHMMModel(K, a, b, alpha, beta)

    # Sample dummy data
    sampled_model = PoissonHMM.sample_model(prior)
    X, S = PoissonHMM.sample_data(5, sampled_model)

    # Inference
    max_iter = 1
    posterior, S_est = PoissonHMM.VI(X, prior, max_iter)

    println(S)
    println(S_est)
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
