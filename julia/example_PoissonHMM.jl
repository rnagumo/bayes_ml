
"""
Example for PoissonHMM (Poisson Hidden Markov Model)
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions
using JLD

push!(LOAD_PATH, ".")
import PoissonHMM

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Hyper-parameters
    K = 2
    a = ones(K)
    b = ones(K)
    alpha = ones(K)
    beta = ones(K, K)
    
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

function test_time_series()
    # Load data
    file_name = "../data/timeseries.jld"
    X = load(file_name)["obs"]

    # Prior
    K = 2
    a = ones(K)
    b = 0.01 * ones(K)
    alpha = 10.0 * ones(K)
    beta = 100.0 * eye(K) + ones(K, K)
    prior = PoissonHMM.PoissonHMMModel(K, a, b, alpha, beta)

    # Inference
    max_iter = 100
    posterior, S_est = PoissonHMM.VI(X, prior, max_iter)
    
    # Visualize
    figure("Poisson HMM")
    subplot(211)
    plot(X)

    subplot(212)
    fill_between(1:length(X), S_est[:, 1])

    try
        show()
    catch
        close()
    end
end

eye(D::Int) = Matrix{Float64}(I, D, D)

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
    func_dict = Dict([("dummy", check_with_dummy_data),
                      ("time", test_time_series)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()
