
module BayesNeuralNet

"""
Bayesian Neural Network (3 layer Perceptron)
"""

using Distributions

struct BayesNNModel
    M::Int  # Dimension of input
    K::Int  # Dimension of latent
    D::Int  # Dimension of output
    sigma2_w::Float64  # variance of w
    sigma2_y::Float64  # variance of y
end

struct NeuralNetModel
    M::Int  # Dimension of input
    K::Int  # Dimension of latent
    D::Int  # Dimension of output
    W1::Array{Float64, 2}  # K*M
    W2::Array{Float64, 2}  # D*K
    sigma2_y::Float64  # variance of y
end

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

function basis_func(x::Array{Float64, 1}, D::Int)
    """
    Basis function that converts scalar to D-dimensional vector

    x -> [1, x, x^2, ..., x^{D-1}]

    Parameters
    ----------
    x : Array of size N
        Input vector 

    D : Int
        Output dimensio

    Returns
    -------
    X : Array of size N*D
        Converted output vector
    """

    X = repeat(x, 1, D) .^ (0:D - 1)'
    return X
end

# -----------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------

function sample_data_from_prior(model::BayesNNModel, x::Array{Float64, 1})
    """
    Sample function given data x
    """

    M = model.M
    K = model.K
    D = model.D
    N = size(x, 1)

    W1 = sqrt(model.sigma2_w) .* randn(K, M)
    W2 = sqrt(model.sigma2_w) .* randn(D, K)

    # Sample function
    X = basis_func(x, M)  # N*M
    Y = tanh.(X * W1') * W2'  # N*D

    # Sample data
    Y_obs = Y + sqrt(model.sigma2_y) .* randn(N, D)

    # Sampled model
    nn_model = NeuralNetModel(M, K, D, W1, W2, model.sigma2_y)

    return Y, Y_obs, nn_model
end

function laprace_approximation(model::BayesNNModel, x::Array{Float64, 1},
                               y::Array{Float64, 2}, lr::Float64=0.01,
                               max_iter::Int=5)
    
    # Dimension
    M = model.M
    K = model.K
    D = model.D
    N = size(x, 1)

    # Inputs
    X = basis_func(x, M)  # N*M

    # Initialize parameters
    W1 = sqrt(model.sigma2_w) .* randn(K, M)
    W2 = sqrt(model.sigma2_w) .* randn(D, K)

    for iter in 1:max_iter
        # Forward
        a1 = X * W1'
        z1 = tanh.(a1)

        a2 = z1 * W2'

        # Backward
        d2 = a2 - y
        dE_dW2 = d2' * z1

        d1 = (1 ./ cosh.(a1) .^ 2) .* (d2 * W2)
        dE_dW1 = d1' * X

        # Gradient
        dp_dW2 = -(1 / model.sigma2_y .* dE_dW2 + 1 / model.sigma2_w .* W2)
        dp_dW1 = -(1 / model.sigma2_y .* dE_dW1 + 1 / model.sigma2_w .* W1)

        # Update Parameters
        W2 = W2 + lr .* dp_dW2
        W1 = W1 + lr .* dp_dW1
    end

end

end
