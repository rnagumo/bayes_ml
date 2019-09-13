
"""
Example for Gauss (Non-negative Matrix Factorization)
"""

using LinearAlgebra
using PyPlot, PyCall
using ArgParse
using Distributions
using DataFrames

push!(LOAD_PATH, ".")
import Gauss

# -----------------------------------------------------------
# Executing function
# -----------------------------------------------------------

eye(D::Int) = Matrix{Float64}(I, D, D)

function check_with_dummy_data()

    # Prior
    D = 2
    m = rand(D)
    beta = 1.0
    nu = 10
    W = eye(D)
    prior = Gauss.GaussModel(D, m, beta, nu, W)

    # Sample dummy data
    sampled_model = Gauss.sample_model(prior)
    X = Gauss.sample_data(5, sampled_model)

    # Inference
    posterior = Gauss.inference(X, prior)

    # Prediction
    X_pred, sampled_posterior = Gauss.sample_prediction(5, posterior)

    println(X)
    println(X_pred)
end

function plot_2d_gauss()
    # -------------------------------------------------------------
    # Data & modeling
    # -------------------------------------------------------------

    # Prior
    D = 2
    m = rand(D)
    beta = 0.01
    nu = 10
    W = eye(D)
    prior = Gauss.GaussModel(D, m, beta, nu, W)

    # Sample dummy data
    sampled_model = Gauss.sample_model(prior)
    N_sample = 200
    X = Gauss.sample_data(N_sample, sampled_model)

    # Inference
    posterior = Gauss.inference(X, prior)

    # Prediction
    N_pred = 20
    X_pred, sampled_posterior = Gauss.sample_prediction(N_pred, posterior)

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    # Prepare x & y axis grid
    R = 200
    xmin = minimum(X[:, 1]) - 5
    xmax = maximum(X[:, 1]) + 5
    ymin = minimum(X[:, 2]) - 5
    ymax = maximum(X[:, 2]) + 5
    x1 = range(xmin, stop=xmax, length=R)
    x2 = range(ymin, stop=ymax, length=R)
    x1grid = repeat(x1, 1, R)
    x2grid = repeat(x2', R, 1)
    X_grid = [x1grid[:] x2grid[:]]

    # Calculate pdf
    Z_prior = pdf(MvNormal(sampled_model.mu, inv(sampled_model.Lambda)),
                  X_grid')
    Z_prior = reshape(Z_prior, R, R)

    Z_posterior = pdf(MvTDist(sampled_posterior.nu, sampled_posterior.mu,
                              sampled_posterior.Lambda),
                      X_grid')
    Z_posterior = reshape(Z_posterior, R, R)

    # Plot
    figure(figsize=(12, 6))

    subplot(121)
    plot(X[:, 1], X[:, 2], "ob", alpha=0.2)
    contour(x1grid, x2grid, Z_prior, alpha=0.9, cmap=get_cmap("bwr"))
    title("Observed samples")

    subplot(122)
    plot(X_pred[:, 1], X_pred[:, 2], "og", alpha=0.2)
    contour(x1grid, x2grid, Z_posterior, alpha=0.9, cmap=get_cmap("PRGn"))
    title("Prediction")

    tight_layout()

    try
        show()
    catch
        close()
    end

end

function gauss_pdf_1s(mu::Array, Sigma::Array, res::Int=100, c::Float64=1.0)
    """
    Return Gaussian pdf (probability density function) in 1 sigma
    """
    # Eigenvalues, eigenvectors
    val, vec = eigen(Sigma)
    w = 2 * pi / res .* (0:res)

    a = sqrt(c * val[1])
    b = sqrt(c * val[2])
    p1 = a .* cos.(w)
    p2 = b .* sin.(w)
    P = repeat(mu', res + 1, 1) + vcat(p1', p2')' * vec

    return P
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
                      ("plot", plot_2d_gauss)])

    # Execute selected function
    func_dict[parsed_args["func"]]()
end

main()
