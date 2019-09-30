
"""
Example for PoissonMixture
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
import GaussianMixture

eye(D::Int) = Matrix{Float64}(I, D, D)

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()
    # Prior
    D = 2
    K = 3
    alpha = 100.0 * ones(K)
    m = 10.0 * ones(K, D)
    beta = 0.1 * ones(K)
    nu = 10.0 * ones(K)
    W = repeat(eye(D), 1, 1, K)
    prior = GaussianMixture.GaussianMixtureModel(D, K, alpha, m, beta, nu, W)

    # Sample data
    N = 10
    sampled_model = GaussianMixture.sample_model(prior)
    X, Z = GaussianMixture.sample_data(N, sampled_model)

    # Inference
    posterior, Z_est = GaussianMixture.variational_inference(X, prior, 1)
end

function test_2d_plot()

    # ---------------------------------------------------------
    # Sample data
    # ---------------------------------------------------------

    # Prior
    D = 2
    K = 3
    alpha = 100.0 * ones(K)
    m = 10.0 * ones(K, D)
    beta = 0.1 * ones(K)
    nu = 10.0 * ones(K)
    W = repeat(eye(D), 1, 1, K)
    prior = GaussianMixture.GaussianMixtureModel(D, K, alpha, m, beta, nu, W)

    # Sample data
    N = 300
    sampled_model = GaussianMixture.sample_model(prior)
    X, Z = GaussianMixture.sample_data(N, sampled_model)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    # Inference
    max_iter = 50
    posterior, Z_est = GaussianMixture.variational_inference(
        X, prior, max_iter)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    # Visualization
    Z_cat = get_argmax(Z)
    Z_est_cat = get_argmax(Z_est)
    colors = ["b", "r", "g", "orange", "c", "grey"]

    figure(figsize=(12, 6))
    subplot(121)
    scatter(X[:, 1], X[:, 2], color=colors[Z_cat])
    title("Truth")

    subplot(122)
    scatter(X[:, 1], X[:, 2], color=colors[Z_est_cat])
    title("Estimation")

    tight_layout()
    try
        show()
    catch
        close()
    end
end

function get_argmax(X::Array)
    N = size(X, 1)
    idx = zeros(Int32, N)
    for n in 1:N
        idx[n] = argmax(X[n, :])
    end

    return idx
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
    func_dict = Dict([("dummy", check_with_dummy_data),
                      ("plot", test_2d_plot),])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()
