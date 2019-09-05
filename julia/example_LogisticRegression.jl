
"""
Example for LogisticRegression
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions
using JLD

push!(LOAD_PATH, ".")
import LogisticRegression

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Prior
    M = 2
    lambda = 100.0    
    prior = LogisticRegression.LogisticRegressionModel(M, lambda)
    sampled_model = LogisticRegression.sample_model(prior)

    # Sample dummy data
    N = 5
    X = 2 .* rand(N, M) .- 1.0  # Input
    Y = LogisticRegression.sample_data(X, sampled_model)

    # Inference
    max_iter = 1
    posterior = LogisticRegression.VI(X, Y, prior)

    println(X, Y)
    # println(S_est)
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
