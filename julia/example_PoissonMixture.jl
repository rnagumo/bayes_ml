
"""
Example for PoissonMixture
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions

push!(LOAD_PATH, ".")
import PoissonMixture

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

function check_with_dummy_data()

    # Hyper-parameters
    D = 2
    K = 3
    alpha = 100.0 * ones(K)
    a = ones(D, K)
    b = 0.01 * ones(K)

    # Prior
    prior = PoissonMixture.PoissonMixtureModel(D, K, alpha, a, b)

    # Sample dummy data
    N = 10
    sampled_model = PoissonMixture.sample_model(prior)
    X, S = PoissonMixture.sample_data(N, sampled_model)

    # Inference
    max_iter = 1
    posterior, S_est = PoissonMixture.VI(X, prior, max_iter)
    # posterior, S_est = PoissonMixture.GS(X, prior, max_iter)
    
    println(S_est)
end

function test_2d_plot()

    # ---------------------------------------------------------
    # Sample data
    # ---------------------------------------------------------

    D = 2
    K = 6
    alpha = 100.0 * ones(K)
    a = ones(D, K)
    b = 0.01 * ones(K)
    prior = PoissonMixture.PoissonMixtureModel(D, K, alpha, a, b)

    # Sample dummy data
    N = 300
    sampled_model = PoissonMixture.sample_model(prior)
    X, S = PoissonMixture.sample_data(N, sampled_model)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    # Inference & Visualization
    max_iter = 100
    posterior, S_est = PoissonMixture.VI(X, prior, max_iter)
    # posterior, S_est = PoissonMixture.GS(X, prior, max_iter)

    # Visualization
    S_cat = get_argmax(S)
    S_est_cat = get_argmax(S_est)
    colors = ["b", "r", "g", "orange", "c", "grey"]

    figure(figsize=(12, 6))
    subplot(121)
    scatter(X[:, 1], X[:, 2], color=colors[S_cat])
    title("Truth")

    subplot(122)
    scatter(X[:, 1], X[:, 2], color=colors[S_est_cat])
    title("Estimation")

    tight_layout()
    try
        show()
    catch
        close()
    end
end

function test_2d_plot_with_save()
    # Prior
    D = 2
    K = 6
    alpha = 100.0 * ones(K)
    a = ones(D, K)
    b = 0.01 * ones(K)    
    prior = PoissonMixture.PoissonMixtureModel(D, K, alpha, a, b)

    # Sample dummy data
    N = 500
    sampled_model = PoissonMixture.sample_model(prior)
    X, S = PoissonMixture.sample_data(N, sampled_model)

    # Inference & Visualization
    max_iter = 100

    # -----------------------------------------------------
    # Variational Inference
    # -----------------------------------------------------

    posterior, S_est = PoissonMixture.VI(X, prior, max_iter)

    # Visualization
    S_cat = get_argmax(S)
    S_est_cat = get_argmax(S_est)
    colors = ["b", "r", "g", "orange", "c", "grey", "k"]

    figure(figsize=(12, 6))
    subplot(121)
    scatter(X[:, 1], X[:, 2], color=colors[S_cat])
    title("Truth")

    subplot(122)
    scatter(X[:, 1], X[:, 2], color=colors[S_est_cat])
    title("Estimation")

    suptitle("Variational Inference")
    tight_layout()
    savefig("../results/pmm_vi.png")
    close()

    # -----------------------------------------------------
    # Variational Inference
    # -----------------------------------------------------
    posterior, S_est = PoissonMixture.GS(X, prior, max_iter)

    # Visualization
    S_cat = get_argmax(S)
    S_est_cat = get_argmax(S_est)

    figure(figsize=(12, 6))
    subplot(121)
    scatter(X[:, 1], X[:, 2], color=colors[S_cat])
    title("Truth")

    subplot(122)
    scatter(X[:, 1], X[:, 2], color=colors[S_est_cat])
    title("Estimation")

    suptitle("Gibbs sampling")
    tight_layout()
    savefig("../results/pmm_gs.png")
    close()
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
                      ("plot", test_2d_plot),
                      ("save", test_2d_plot_with_save)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()
